"""Extended network builder that treats both public and private schools as origins.

This module provides :class:`MixedOriginNetworkBuilder`, a subclass of
:class:`OptimizedSchoolNetworkBuilder`.  The new builder processes public and
private schools in the same run so that distance matrices include routes from
all school types.  Use it when both public and private schools should act as
origin points.

Example
-------
::

    from modules.mixed_origin_network_builder import MixedOriginNetworkBuilder
    builder = MixedOriginNetworkBuilder(
        G_road,
        gdf_public,
        gdf_private,
        gdf_peripheral,
        admin_gdf,
        num_processes=4,
    )
    results = builder.build_complete_network()
"""

from modules.optimized_network_builder import (
    OptimizedSchoolNetworkBuilder,
    _init_worker,
    _process_school_worker,
)

__all__ = ["MixedOriginNetworkBuilder"]


class MixedOriginNetworkBuilder(OptimizedSchoolNetworkBuilder):
    """Network builder that uses both public and private schools as origins."""

    def _parallel_process_schools(self, buffer_distance_m, max_distance_km):
        """Process origin schools from both public and private datasets."""
        from multiprocessing import get_context, cpu_count

        origin_schools = (
            self.public_schools["school_id"].tolist()
            + self.private_schools["school_id"].tolist()
        )

        print(f"🔄 Processing {len(origin_schools)} schools in parallel...")

        ctx = get_context("fork")
        num_workers = self.num_processes or cpu_count()
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(self,),
        ) as pool:
            results = pool.starmap(
                _process_school_worker,
                [
                    (sid, buffer_distance_m, max_distance_km)
                    for sid in origin_schools
                ],
            )

        return results

    def _process_single_school_optimized(
        self, school_id, buffer_distance_m, max_distance_km
    ):
        """Compute candidate routes for one origin school regardless of type."""
        try:
            if school_id in self.public_schools["school_id"].values:
                school_row = self.public_schools[
                    self.public_schools["school_id"] == school_id
                ].iloc[0]
            else:
                school_row = self.private_schools[
                    self.private_schools["school_id"] == school_id
                ].iloc[0]

            nearby_schools = self._find_nearby_schools_fast(
                school_row.geometry, buffer_distance_m
            )

            distance_matrix = self._calculate_distances_fast(
                school_id, nearby_schools
            )

            if max_distance_km:
                distance_matrix = distance_matrix.where(
                    distance_matrix <= max_distance_km * 1000, np.nan
                )

            candidates = self._get_candidates_fast(distance_matrix)
            routes = self._calculate_routes_fast(school_id, candidates)
            return school_id, distance_matrix.to_dict(), routes

        except Exception as e:
            print(f"Error processing school {school_id}: {e}")
            return school_id, {}, {}

