from collections.abc import Iterable
from functools import lru_cache
from typing import Sequence

import numpy as np
from bg_atlasapi import BrainGlobeAtlas
from bg_space_extra import AnatomicalPoints, AnatomicalStack
from brainglobe_space import AnatomicalSpace


class Atlas(BrainGlobeAtlas):
    """Wrapper for bg_atlasapi BrainGlobeAtlas

    Class Attributes
    ----------------
    midline : int
        Position of the midline on the left-right axis.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas to be used.
    **kwargs
        Keyword arguments passed to BrainGlobeAtlas
    """

    space: AnatomicalSpace
    midline = 282
    # structure_blacklist: Tuple[int, ...] = (0, 805, 808, 812, 842, 864, 872, 881, 898)

    def __init__(self, atlas_name="mpin_zfish_1um", check_latest=False, **kwargs):
        super().__init__(atlas_name=atlas_name, check_latest=check_latest, **kwargs)
        self._annotation = AnatomicalStack(self.space, self.annotation)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            idx, key_ = key

            if isinstance(idx, Iterable) and not isinstance(idx, str):
                return [self.structures[i][key_] for i in idx]

            return self.structures[idx][key_]
        else:
            if isinstance(key, Iterable) and not isinstance(key, str):
                return [self.structures[i] for i in key]

            return self.structures[idx]

    def _to_region_id_list(self, regions):
        if isinstance(regions, Iterable):
            if isinstance(regions, str):
                return self[[regions], "id"]
            else:
                return self[regions, "id"]
        else:
            return self[[regions], "id"]

    def points_to_structures(
        self, points: AnatomicalPoints, out_of_range_value=0, return_type="id"
    ):
        """Convert anatomical points to structures

        Parameters
        ----------
        points : AnatomicalPoints
            Anatomical points.
        return_type : str
            Return type (e.g., "id", "name")

        Returns
        -------
        ndarray of shape (n,)
            Structure IDs.
        """
        structure_ids = self.annotation.sel(
            points, out_of_range_value=out_of_range_value
        )

        if return_type == "id":
            return structure_ids

        return np.array([self.structures[i][return_type] for i in structure_ids])

    def is_points_in_structures(
        self,
        points: AnatomicalPoints,
        structures: Sequence[str],
        include_descendants=True,
        out_of_range_value=0,
    ):
        """Check whether points are inside the specified structures.

        Parameters
        ----------
        points : AnatomicalPoints
            Coordinates of points in anatomical space of the atlas, or structures of points.
        structures : sequence of str
            structures to check for.
        include_descendants : bool
            Whether to include descendants of structures.
        Returns
        -------
        ndarray of shape (n,)
            Whether points are inside the specified structures.
        """
        structures = self._to_region_id_list(structures)

        if include_descendants:
            structure_ids = self.get_structures_descendants(
                structures, include_self=True, key="id"
            )
        else:
            structure_ids = np.unique([self.structures[i]["id"] for i in structures])

        point_structure_ids = self.points_to_structures(
            points, out_of_range_value=out_of_range_value
        )

        return np.isin(point_structure_ids, structure_ids)

    def is_in_structures(self, structure_ids, structures):
        return np.isin(structure_ids, self.get_structure_descendants(structures))

    def get_structures_descendants(self, structures, include_self=True, key="id"):
        structures = self._to_region_id_list(structures)

        descendants = list(structures) if include_self else []

        for structure in structures:
            descendants.extend(self.get_structure_descendants(structure))

        return np.unique(self[descendants, key])

    def get_structures_stack(self, include, exclude=(), target_space="sal"):
        orig_include_ids = self[include, "id"]
        include_ids = self.get_structures_descendants(include, include_self=True)
        exclude_ids = self.get_structures_descendants(exclude, include_self=True)
        exclude_ids = np.setdiff1d(exclude_ids, orig_include_ids)
        structure_ids = np.setdiff1d(include_ids, exclude_ids)
        structure_ids = structure_ids[structure_ids != 0]

        mask = AnatomicalStack(
            self.annotation.space, np.isin(self.annotation, structure_ids)
        ).asspace(target_space)

        return mask.asspace(target_space)

    @lru_cache
    def get_masks(
        self,
        include,
        exclude=(),
        method="max",
        views=("top", "front", "left"),
    ):
        spaces = dict(top="sal", front="asl", left="las")

        single_view = False

        if isinstance(views, str):
            views = (views,)
            single_view = True

        orig_include_ids = self[include, "id"]
        include_ids = self.get_structures_descendants(include, include_self=True)
        exclude_ids = self.get_structures_descendants(exclude, include_self=True)
        exclude_ids = np.setdiff1d(exclude_ids, orig_include_ids)
        structure_ids = np.setdiff1d(include_ids, exclude_ids)
        structure_ids = structure_ids[structure_ids != 0]

        if isinstance(method, str):
            from hashlib import sha1
            from pathlib import Path

            bs = " ".join(map(str, (*sorted(structure_ids), method))).encode()
            s = np.base_repr(int.from_bytes(sha1(bs).digest(), "big"), 36).lower()
            mask_dir = Path().home() / ".analysis_bundle/brain_mask"
            mask_path = (mask_dir / s).with_suffix(".npz")

            if mask_path.exists():
                masks = dict(np.load(mask_path.as_posix()))
            else:
                mask = AnatomicalStack(
                    self.annotation.space, np.isin(self.annotation, structure_ids)
                )
                masks = {
                    name: getattr(np.asarray(mask.asspace(space)), method)(axis=0)
                    for name, space in spaces.items()
                }

                mask_path.parent.mkdir(exist_ok=True, parents=True)
                np.savez_compressed(mask_path, **masks)
        else:
            mask = AnatomicalStack(
                self.annotation.space, np.isin(self.annotation, structure_ids)
            )
            masks = {
                view: method(np.asarray(mask.asspace(spaces[view])), axis=0)
                for view in views
            }

        if single_view:
            return masks[views[0]]
        else:
            return {view: masks[view] for view in views}
