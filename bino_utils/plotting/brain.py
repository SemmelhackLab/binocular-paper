from collections.abc import Iterable
from functools import cached_property

import cv2
import matplotlib.pyplot as plt
import numpy as np
from bg_space_extra import AnatomicalPoints
from matplotlib.patches import Polygon
from mplex import Grid
from mplex.axes import add_bounding_axes
from mplex.utils import safe_unpack
from scipy.ndimage import gaussian_filter1d

from bino_utils.atlas import Atlas
from bino_utils import get_bbox


def add_anatomical_axes(
    xy,
    dxy,
    text=("L", "A"),
    text_pad=2,
    loc="upper left",
    c="k",
    lw=0.5,
    ax=None,
    annotate_kw=None,
    **plot_kw,
):
    from mplex.utils import safe_unpack

    if ax is None:
        ax = plt.gca()

    if annotate_kw is None:
        annotate_kw = dict()

    dxy = safe_unpack(dxy)
    text = safe_unpack(text)
    text_pad = safe_unpack(text_pad)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    loc = loc.lower()

    x0, y0 = xy

    x1 = x0 + dxy[0]
    y1 = y0 + dxy[1]

    ha = "left" if (dxy[0] > 0) == (xlim[1] > xlim[0]) else "right"
    va = "baseline" if (dxy[1] > 0) == (ylim[1] > ylim[0]) else "top"
    xlabel = ax.annotate(
        text[0],
        (x1, y0),
        (text_pad[0], 0),
        textcoords="offset points",
        va="center",
        ha=ha,
        annotation_clip=False,
    )
    ylabel = ax.annotate(
        text[1],
        (x0, y1),
        (0, text_pad[1]),
        textcoords="offset points",
        ha="center",
        va=va,
        annotation_clip=False,
    )

    xline = ax.plot([x0, x0], [y0, y1], c=c, lw=lw, **plot_kw)[0]
    yline = ax.plot([x0, x1], [y0, y0], c=c, lw=lw, **plot_kw)[0]

    return dict(xline=xline, yline=yline, xlabel=xlabel, ylabel=ylabel)


def get_projs(stack, method="mean", bounds=None):
    if bounds is None:
        bounds = {}
    spaces = dict(top="sal", front="asl", left="las")
    projs = {
        name: getattr(
            np.asarray(stack.asspace(space))[bounds.get(space[0], np.s_[:])], method
        )(axis=0)
        for name, space in spaces.items()
    }
    return projs


def add_mask_as_polygon(
    mask,
    sigma=5,
    n=None,
    fc="none",
    ec="k",
    lw=0.5,
    alpha=1,
    zorder=1000,
    clip_on=True,
    ax=None,
    **kwargs,
):
    kwargs = dict(
        dict(fc=fc, ec=ec, lw=lw, clip_on=clip_on, zorder=zorder, alpha=alpha), **kwargs
    )
    if ax is None:
        ax = plt.gca()
    cnts = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    if n is not None:
        cnts = cnts[:n]

    for cnt in cnts:
        xy = cnt[:, 0]
        ax.add_patch(
            Polygon(gaussian_filter1d(xy, sigma, 0) if sigma else xy, **kwargs)
        )


class BrainPlot:
    def __init__(
        self, atlas: Atlas, top=None, front=None, left=None, empty=None, bounds=None
    ):
        self.atlas = atlas
        self.axes_dict = {}
        self.bounds = bounds

        if top is not None:
            self.axes_dict["top"] = top
        if front is not None:
            self.axes_dict["front"] = front
        if left is not None:
            self.axes_dict["left"] = left
        self.empty = empty

    def __getitem__(self, key):
        return self.axes_dict[key]

    def make_ax(self, **kwargs):
        from mplex.axes_collection import AxArray

        return AxArray(self.axes_dict.values()).make_ax(**kwargs)

    def add_anatomical_axes(
        self,
        xl=None,
        xr=None,
        yt=None,
        yb=None,
        length=50,
        fontsize=6,
        align="bottom",
        **kw,
    ):
        ret = dict()

        if xl is None:
            xl = self["top"].get_xlim()[0]

        if xr is None:
            xr = self["left"].get_xlim()[0]

        if yt is None:
            yt = self["front"].get_ylim()[0]

        if yb is None:
            yb = self["top"].get_ylim()[0]

        if "top" in self.axes_dict:
            ret["top"] = add_anatomical_axes(
                (xl, yb),
                -length,
                ("L", "A"),
                ax=self["top"],
                loc="lower left" if align == "bottom" else "upper left",
                clip_on=False,
                text_pad=1,
                annotate_kw=dict(size=fontsize),
                **kw,
            )

        if "front" in self.axes_dict:
            ret["front"] = add_anatomical_axes(
                (xl, yt),
                -length,
                ("L", "D"),
                ax=self["front"],
                clip_on=False,
                loc="lower left" if align == "bottom" else "upper left",
                text_pad=1,
                annotate_kw=dict(size=fontsize),
                **kw,
            )

        if "left" in self.axes_dict:
            ret["left"] = add_anatomical_axes(
                (xr, yb),
                -length,
                ("D", "A"),
                ax=self["left"],
                loc="lower left" if align == "bottom" else "upper right",
                clip_on=False,
                text_pad=1,
                annotate_kw=dict(size=fontsize),
                **kw,
            )
        return ret

    @cached_property
    def bounding_ax(self):
        ax = add_bounding_axes(*self.axes_dict.values())
        ax.axis("off")
        return ax

    def scatter_values(
        self,
        points: AnatomicalPoints,
        values,
        lw=0,
        mode="c",
        cmap="hot",
        s=5,
        vmin=None,
        vmax=None,
        **kwargs,
    ):
        points = points.asspace(self.atlas.space)
        values = np.asarray(values)

        if self.bounds is not None:
            sel = np.ones(len(points), bool)
            for c1, c2 in zip("sal", "ipr"):
                if c1 in self.bounds:
                    sel &= points[c1] >= self.bounds[c1]
                if c2 in self.bounds:
                    sel &= points[c1] <= self.bounds[c2]
            points = points[sel]
            values = values[sel]

        for view, x, y, z in zip(("top", "front", "left"), "lls", "asa", "sal"):
            if view in self.axes_dict:
                if mode == "c":
                    argsort = points[z].argsort()[::-1]
                elif mode == "f":
                    argsort = points[z].argsort()
                elif mode == "a":
                    argsort = values.argsort()[::-1]
                else:
                    argsort = values.argsort()

                px, py, c = points[x][argsort], points[y][argsort], values[argsort]

                self.axes_dict[view].scatter(
                    px, py, c=c, cmap=cmap, vmin=vmin, vmax=vmax, lw=lw, s=s, **kwargs
                )

    def scatter(self, points: AnatomicalPoints, lw=0, s=5, **kwargs):
        points = points.asspace(self.atlas.space)

        for view, x, y in zip(("top", "front", "left"), "lls", "asa"):
            if view in self.axes_dict:
                self.axes_dict[view].scatter(points[x], points[y], lw=lw, s=s, **kwargs)

    def set_title(self, *args, **kwargs):
        return self.bounding_ax.set_title(*args, **kwargs)

    def add_masks(self, masks, sigma=0, **kwargs):
        for view, ax in self.axes_dict.items():
            mask = masks[view]
            if mask.dtype == bool:
                add_mask_as_polygon(masks[view], sigma=sigma, ax=ax, **kwargs)
            else:
                ax.imshow(mask, **kwargs)

    def add_structures(
        self,
        *structures,
        exclude=("peripheral nervous system", "retina"),
        method="max",
        sigma=5,
        **kwargs,
    ):
        views = tuple(self.axes_dict)
        masks = self.atlas.get_masks(
            structures, exclude=exclude, method=method, views=views
        )
        return self.add_masks(masks, sigma=sigma, **kwargs)

    def add_projs(
        self, stack, method="mean", vmin=None, vmax=None, cmap="gray_r", bounds=None
    ):
        if bounds is None:
            bounds = {}
        projs = get_projs(stack, method, bounds)
        for view in self.axes_dict:
            self[view].imshow(projs[view], cmap=cmap, vmin=vmin, vmax=vmax)


class BrainGrid:
    def __init__(
        self,
        atlas: Atlas,
        shape=(1, 1),
        structures="root",
        exclude=("peripheral nervous system", "retina"),
        w=None,
        h=None,
        space_within=2,
        space_across=10,
        space_border=(0, 0, 0, 0),
        mask_kw=None,
        fc="none",
        front=True,
        left=True,
        bounds=None,
        pad=2,
        **kwargs,
    ):
        self.front = front
        self.left = left

        if isinstance(structures, str) or not isinstance(structures, Iterable):
            structures = (structures,)

        masks = atlas.get_masks(structures, exclude=exclude, method="max")
        bboxes = self.maximize_bbox(
            **{name: get_bbox(mask) for name, mask in masks.items()}
        )

        for view, arr in bboxes.items():
            arr += (-pad, pad)

        temp = dict(
            a=dict(top=(0, 0), left=(0, 0)),
            p=dict(top=(0, 1), left=(0, 1)),
            l=dict(top=(1, 0), front=(1, 0)),
            r=dict(top=(1, 1), front=(1, 1)),
            s=dict(front=(0, 0), left=(1, 0)),
            i=dict(front=(0, 1), left=(1, 1)),
        )

        if bounds is None:
            bounds = dict()

        for k0, v0 in temp.items():
            if k0 in bounds:
                for k1, v1 in v0.items():
                    bboxes[k1][v1] = bounds[k0]

        nrows, ncols = safe_unpack(shape, default1=1)

        w1 = bboxes["top"][1].ptp()
        h2 = bboxes["top"][0].ptp()

        if self.left:
            w2 = bboxes["left"][1].ptp()
            w_ = [w1, w2]
        else:
            w_ = [w1]

        if self.front:
            h1 = bboxes["front"][0].ptp()
            h_ = [h1, h2]
        else:
            h_ = [h2]

        gridsize = [np.tile(w_, ncols), np.tile(h_, nrows)]

        assert not ((h is not None) and (w is not None))

        if isinstance(space_within, (int, float)):
            space_within = (space_within, space_within)

        if isinstance(space_across, (int, float)):
            space_across = (space_across, space_across)

        space_y = space_within[1] * self.front

        if w is not None:
            space_x = space_within[0] * self.left
            gridsize = [i / sum(w_) * (w - space_x) for i in gridsize]
        elif h is not None:
            space_y = space_within[1] * self.front
            gridsize = [i / sum(h_) * (h - space_y) for i in gridsize]

        nrows_cell = int(self.front) + 1
        ncols_cell = int(self.left) + 1

        space = (
            np.tile((space_within[0], space_across[0])[-ncols_cell:], ncols)[:-1],
            np.tile((space_within[1], space_across[1])[-nrows_cell:], nrows)[:-1],
        )
        space = (
            np.concatenate([[space_border[0]], space[0], [space_border[1]]]),
            np.concatenate([[space_border[2]], space[1], [space_border[3]]]),
        )

        self._grid = Grid(
            axsize=gridsize,
            shape=(nrows * nrows_cell, ncols * ncols_cell),
            space=space,
            sharex="c",
            sharey="r",
            facecolor=fc,
            **kwargs,
        )

        self.axes_by_view = dict()

        axes = self._grid.axs

        if self.front and not self.left:
            self.axes_by_view = dict(top=axes[1::2], front=axes[::2])
        elif self.left and not self.front:
            self.axes_by_view = dict(top=axes[:, ::2], left=axes[:, 1::2])
        elif self.left and self.front:
            self.axes_by_view = dict(
                top=axes[1::2, ::2],
                front=axes[::2, ::2],
                left=axes[1::2, 1::2],
                empty=axes[::2, 1::2],
            )
        else:
            self.axes_by_view = dict(top=axes)

        for view, axes in self.axes_by_view.items():
            if view in bboxes:
                for ax in axes.ravel():
                    ax.set_xlim(bboxes[view][1])
                    ax.set_ylim(bboxes[view][0])

        for ax in self._grid.axes.ravel():
            ax.yaxis.set_inverted(True)

        if "left" in self.axes_by_view:
            for ax in self.axes_by_view["left"].ravel():
                ax.xaxis.set_inverted(True)

        self._grid.axis("off")
        self.brain_plots = np.apply_along_axis(
            lambda x: BrainPlot(
                atlas=atlas, bounds=bounds, **dict(zip(tuple(self.axes_by_view), x))
            ),
            -1,
            np.stack(tuple(self.axes_by_view.values()), axis=-1),
        )

        if mask_kw is None:
            mask_kw = {}

        self._cp = self[:]

    def __getattr__(self, name):
        if isinstance(self._cp, BrainPlot):
            return getattr(self._cp, name)

        a = np.array([getattr(ax, name) for ax in self._cp.ravel()], object).reshape(
            (*self._cp.shape, -1)
        )

        if all(map(callable, a.ravel())):
            return lambda *args, **kwargs: np.array(
                [i(*args, **kwargs) for i in a.ravel()]
            ).reshape((*self._cp.shape, -1))

        return a

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.axes_by_view[key]
        else:
            return self.brain_plots[key]

    @property
    def grid(self):
        return self._grid

    @property
    def nrows(self):
        return self.brain_plots.shape[0]

    @property
    def ncols(self):
        return self.brain_plots.shape[1]

    def maximize_bbox(self, top, front, left):
        top, front, left = top.copy(), front.copy(), left.copy()

        if self.front:
            top[1] = front[1] = np.quantile(
                (top[1], front[1]), (0, 1)
            )  # maximize width

        if self.left:
            top[0] = left[0] = np.quantile((top[0], left[0]), (0, 1))  # maximize height

        return dict(top=top, front=front, left=left)

    @property
    def cp(self):
        """Current plot"""
        return self._cp

    def scp(self, *keys):
        """Set current plot"""
        self._cp = self[keys]
