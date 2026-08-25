"""
Automated truck-routing distances via OpenStreetMap, as a replacement for the
manual ArcGIS Network Analyst workflow a colleague used to produce the 21
arcs currently in node_metrics_150.xlsx's 'truck' sheet (routing on the road
network while excluding tertiary/smaller streets not suitable for HGVs).

Approach:
    1. For each node pair, download just the OSM road network in a tight
       bounding box around that pair (not the whole region at once - see
       "Why per-arc" below), restricted via a custom Overpass filter to
       "truck-suitable" road classes (motorway/trunk/primary/secondary and
       their _link ramps) - i.e. excluding tertiary, residential,
       unclassified, service, track, etc., matching the colleague's ArcGIS
       exclusion.
    2. Snap both endpoints to their nearest graph node.
    3. Run Dijkstra (edge weight = OSM way length, metres) between them.
    4. Compare against the 21 arcs the colleague already computed in ArcGIS
       (node_metrics_150.xlsx's 'truck' sheet) as a validation check before
       trusting this for any node pair the colleague didn't already do.

Why per-arc instead of one regional download: downloading the OSM road
network for the whole ~760x275 km case-study bbox in one call needs Overpass
to auto-split it into ~74 sub-queries, and in this environment that
consistently failed outright (ConnectTimeout on every one of several
retries) even after fixing osmnx's DNS-pinning bug (see the monkeypatch
below) - likely the public Overpass instance throttling/dropping a client
making that many sequential requests in a short window. A single small
bbox (e.g. the 596-node/959-edge test tile used to confirm the DNS fix)
downloads in ~3 seconds without issue, so this script downloads one
tight bbox per arc instead: slower in wall-clock time across many arcs, but
each request is small and independent, so one failure doesn't cascade and
lose progress on arcs already computed.

This script deliberately does NOT write back into node_metrics_150.xlsx -
after the full pipeline-mirrored truck connectivity made the MILP
unsolvable, expanding truck connectivity is a decision for a human to make
deliberately (e.g. a curated subset), not something to auto-apply. Run this
module directly to print the validation report and optionally save a full
computed distance matrix to a CSV for manual review.

Requires: osmnx (pip install osmnx). Downloads live OSM data - needs
internet access.
"""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

# osmnx pins Overpass DNS resolution to a single IP for the whole process
# (via socket.gethostbyname, which returns only one address of possibly
# several) to keep its rate-limit pausing consistent across a session (see
# osmnx._http._config_dns's docstring). In this environment that pinned IP
# is intermittently unreachable at the raw TCP level even though normal
# multi-address resolution (what a plain `requests` call uses, and what
# osmnx would use without this monkeypatch) reaches a working address for
# the same hostname - confirmed by a plain `requests.get` succeeding while
# a direct socket connect to the gethostbyname()-resolved IP consistently
# timed out. Disabling the pin trades osmnx's within-session rate-limit
# consistency for actually being able to connect.
import osmnx._http as _osmnx_http
_osmnx_http._config_dns = lambda url: None

# Road classes an HGV/CO2-tanker truck can realistically use - motorway,
# trunk (dual-carriageway A-roads) and primary/secondary roads, plus their
# link/ramp variants. Excludes tertiary and everything smaller (residential,
# unclassified, service, track, living_street, path) - matching "excluding
# the smallest streets (tertiary?) that are not suitable for trucks" per the
# colleague's ArcGIS methodology.
TRUCK_SUITABLE_HIGHWAY_CLASSES = [
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
]

CUSTOM_FILTER = '["highway"~"{}"]'.format("|".join(TRUCK_SUITABLE_HIGHWAY_CLASSES))

# The case-study bbox spans most of Northern Italy, which the default
# max_query_area_size splits into ~74 small sub-queries against the public
# Overpass instance - slower (one HTTP round trip + rate-limit pause per
# tile) but each individual request is small/quick, which matters once the
# DNS pin above is disabled and every request re-resolves independently.
# download_truck_road_network() also retries the whole call a few times on
# a connection-level failure (osmnx itself only auto-retries HTTP 429/504,
# not TCP-level connect/read timeouts). Caching is on by default
# (ox.settings.use_cache), so a retry resumes from already-fetched tiles
# rather than re-downloading them.
ox.settings.requests_timeout = 120

# overpass-api.de has been consistently timing out at the TCP-connect level
# from this environment across many runs (while a plain request to a
# different host succeeds), even though it occasionally does go through -
# so download_od_subgraph cycles through these mirrors on retry rather than
# hammering the same (possibly environment-blocked) host three times in a
# row.
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
]

DEFAULT_NODE_METRICS = Path(__file__).resolve().parent.parent.parent / "italy_data" / "geographical_feature" / "node_metrics_150.xlsx"

# Padding around a single arc's two endpoints, as a fraction of their own
# lon/lat span, floored at a minimum in degrees - gives short hops enough
# local road network to snap onto and route through, without ballooning a
# long arc's bbox to (much) more than its own endpoint-to-endpoint extent.
PADDING_FRAC = 0.15
MIN_PADDING_DEG = 0.1


def load_nodes(node_metrics_path: Path = DEFAULT_NODE_METRICS) -> pd.DataFrame:
    """Load the case-study nodes (node_id, node_name, longitude, latitude)."""
    nodes = pd.read_excel(node_metrics_path, sheet_name="nodes", index_col=0)
    nodes.index.name = "node_id"
    # A node_id can be shared by co-located facility rows (e.g. Waste + Cement
    # at the same site) - collapse to one row per physical location for routing.
    nodes = nodes.groupby(nodes.index).agg(
        node_name=("node_name", "first"),
        longitude=("longitude", "first"),
        latitude=("latitude", "first"),
    )
    return nodes


def _od_bbox(from_lon: float, from_lat: float, to_lon: float, to_lat: float) -> tuple[float, float, float, float]:
    lon_span = abs(to_lon - from_lon)
    lat_span = abs(to_lat - from_lat)
    pad_lon = max(MIN_PADDING_DEG, PADDING_FRAC * lon_span)
    pad_lat = max(MIN_PADDING_DEG, PADDING_FRAC * lat_span)
    west, east = min(from_lon, to_lon) - pad_lon, max(from_lon, to_lon) + pad_lon
    south, north = min(from_lat, to_lat) - pad_lat, max(from_lat, to_lat) + pad_lat
    return west, south, east, north


def download_od_subgraph(from_lon: float, from_lat: float, to_lon: float, to_lat: float,
                          max_attempts: int = 3) -> nx.MultiDiGraph | None:
    """
    Download the OSM road network (restricted to TRUCK_SUITABLE_HIGHWAY_CLASSES)
    in a tight bounding box around one O-D pair. Retries on connection-level
    failures (osmnx's own retry logic only covers HTTP 429/504, not
    TCP-level connect/read timeouts). Returns None (rather than raising) if
    every attempt fails, so a batch of many arcs can skip one bad arc
    instead of aborting entirely.
    """
    west, south, east, north = _od_bbox(from_lon, from_lat, to_lon, to_lat)
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        ox.settings.overpass_url = OVERPASS_MIRRORS[(attempt - 1) % len(OVERPASS_MIRRORS)]
        try:
            return ox.graph_from_bbox((west, south, east, north), custom_filter=CUSTOM_FILTER, simplify=True)
        except Exception as e:
            last_exc = e
            print(f"    Attempt {attempt}/{max_attempts} via {ox.settings.overpass_url} failed "
                  f"({e.__class__.__name__}: {e})")
    print(f"    Giving up on bbox ({west:.3f}, {south:.3f}, {east:.3f}, {north:.3f}): {last_exc}")
    return None


def route_distance_km(G: nx.MultiDiGraph, from_lon: float, from_lat: float,
                       to_lon: float, to_lat: float) -> float | None:
    """Shortest-path road distance (km) between two points on graph G, or None if unreachable."""
    from_node, to_node = ox.distance.nearest_nodes(G, X=[from_lon, to_lon], Y=[from_lat, to_lat])
    try:
        length_m = nx.shortest_path_length(G, from_node, to_node, weight="length")
    except nx.NetworkXNoPath:
        # A large bbox gets split into many Overpass sub-queries; one of them
        # silently failing (without osmnx raising) leaves a gap in the graph
        # that looks like genuine unreachability but isn't - flag it via the
        # component count rather than trusting a single "no path" result, so
        # a false negative like this doesn't get locked into the checkpoint
        # (confirmed happening in practice: a re-download of a pair
        # checkpointed as unreachable came back as one 28k-node connected
        # component with a valid route).
        n_components = nx.number_weakly_connected_components(G)
        if n_components > 1:
            print(f"    WARNING: no path found and graph has {n_components} disconnected "
                  f"components - this bbox may have been partially downloaded; treat this "
                  f"result as suspect, not a confirmed unreachability")
        return None
    return round(length_m / 1000.0, 2)


def compute_distances_for_pairs(nodes: pd.DataFrame, pairs: list[tuple[int, int]],
                                 checkpoint_path: Path | None = None) -> pd.DataFrame:
    """
    OSM truck-suitable-road distance (km) for each (from_node_id, to_node_id)
    pair, downloading a small bounding box per pair (see module docstring
    for why). Skips (with a warning, not a crash) any pair whose download or
    routing fails.

    If checkpoint_path is given, results are appended to it after every
    single pair (not just at the end) and already-checkpointed pairs are
    skipped on the way in - this run has been killed mid-way by
    session/environment interruptions unrelated to the script itself more
    than once, and each pair can take a while (large arcs need several
    Overpass sub-queries), so losing an entire run's progress to an
    unrelated interruption is expensive. Re-running with the same
    checkpoint_path simply resumes.
    """
    already_done = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        prior = pd.read_csv(checkpoint_path)
        already_done = {(int(r.from_node), int(r.to_node)): r.osm_km for r in prior.itertuples()}
        print(f"  Resuming from checkpoint: {len(already_done)} pair(s) already computed in {checkpoint_path}")

    rows = []
    for i, (from_id, to_id) in enumerate(pairs):
        from_row, to_row = nodes.loc[from_id], nodes.loc[to_id]
        print(f"  [{i + 1}/{len(pairs)}] {from_row.node_name} (#{from_id}) -> {to_row.node_name} (#{to_id})")

        if (from_id, to_id) in already_done:
            d_km = already_done[(from_id, to_id)]
            print(f"    (from checkpoint) {d_km}")
        else:
            G = download_od_subgraph(from_row.longitude, from_row.latitude, to_row.longitude, to_row.latitude)
            if G is None:
                # Download failed (connectivity, not a real "no route exists"
                # result) - leave it OUT of the checkpoint so it's retried
                # next run, rather than locking in a false NaN forever.
                d_km = np.nan
                should_checkpoint = False
            else:
                d_km = route_distance_km(G, from_row.longitude, from_row.latitude, to_row.longitude, to_row.latitude)
                should_checkpoint = True
                if d_km is None:
                    d_km = np.nan
                    print(f"    No path found on the truck-suitable network within this bbox")
                else:
                    print(f"    {d_km:.2f} km")
            if checkpoint_path is not None and should_checkpoint:
                pd.DataFrame([{"from_node": from_id, "to_node": to_id, "osm_km": d_km}]).to_csv(
                    checkpoint_path, mode="a", header=not checkpoint_path.exists(), index=False
                )

        rows.append({"from_node": from_id, "to_node": to_id, "osm_km": d_km})
    return pd.DataFrame(rows)


def load_reference_truck_distances(node_metrics_path: Path = DEFAULT_NODE_METRICS) -> pd.DataFrame:
    """The colleague's existing ArcGIS-computed truck arcs, long format."""
    truck = pd.read_excel(node_metrics_path, sheet_name="truck", index_col=0)
    truck.columns = truck.columns.astype(int)
    rows = []
    for f in truck.index:
        for t in truck.columns:
            v = truck.loc[f, t]
            if pd.notna(v) and v > 0:
                rows.append({"from_node": f, "to_node": t, "arcgis_km": float(v)})
    return pd.DataFrame(rows)


def validate_against_reference(osm_distances: pd.DataFrame, node_metrics_path: Path = DEFAULT_NODE_METRICS,
                                node_names: dict | None = None) -> pd.DataFrame:
    """
    Compare computed OSM-routed distances (from compute_distances_for_pairs,
    columns from_node/to_node/osm_km) against the colleague's ArcGIS-computed
    arcs for the same node pairs. Returns the comparison table and prints
    summary error stats.
    """
    ref = load_reference_truck_distances(node_metrics_path)
    ref = ref.merge(osm_distances, on=["from_node", "to_node"], how="left")
    ref["diff_km"] = ref["osm_km"] - ref["arcgis_km"]
    ref["pct_diff"] = 100 * ref["diff_km"] / ref["arcgis_km"]

    if node_names:
        ref["from_name"] = ref["from_node"].map(node_names)
        ref["to_name"] = ref["to_node"].map(node_names)
        ref = ref[["from_name", "from_node", "to_name", "to_node", "arcgis_km", "osm_km", "diff_km", "pct_diff"]]

    valid = ref.dropna(subset=["osm_km"])
    print("\n" + "=" * 80)
    print("VALIDATION vs. colleague's ArcGIS-computed truck arcs")
    print("=" * 80)
    print(f"Reference arcs: {len(ref)}   Matched (routable on OSM filtered network): {len(valid)}")
    if len(valid):
        print(f"Mean abs % diff:   {valid['pct_diff'].abs().mean():.1f}%")
        print(f"Median abs % diff: {valid['pct_diff'].abs().median():.1f}%")
        print(f"Max abs % diff:    {valid['pct_diff'].abs().max():.1f}%")
        corr = valid["arcgis_km"].corr(valid["osm_km"])
        print(f"Correlation (ArcGIS vs OSM):  {corr:.4f}")
    unmatched = ref[ref["osm_km"].isna()]
    if len(unmatched):
        print(f"\n{len(unmatched)} reference arc(s) had no path on the filtered OSM network:")
        print(unmatched.to_string(index=False))

    return ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-metrics", type=Path, default=DEFAULT_NODE_METRICS)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "truck_distances_osm_computed.csv",
                         help="Where to save the computed distances (long format CSV)")
    parser.add_argument("--checkpoint", type=Path,
                         default=Path(__file__).resolve().parent / "truck_routing_checkpoint.csv",
                         help="Resumable per-pair progress file - re-running with the same path "
                              "skips pairs already computed")
    args = parser.parse_args()

    nodes = load_nodes(args.node_metrics)
    ref = load_reference_truck_distances(args.node_metrics)
    pairs = list(zip(ref["from_node"], ref["to_node"]))

    print(f"Computing OSM truck-suitable-road distances for the {len(pairs)} arcs the colleague "
          f"already computed in ArcGIS (validation set) ...")
    osm_distances = compute_distances_for_pairs(nodes, pairs, checkpoint_path=args.checkpoint)

    report = validate_against_reference(osm_distances, args.node_metrics, node_names=nodes["node_name"].to_dict())
    report.to_csv(args.out, index=False)
    print(f"\nComparison table saved to {args.out}")


if __name__ == "__main__":
    main()
