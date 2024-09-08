import matplotlib.pyplot as plt
from mplex.text import get_text_bbox


def add_anat_axes(length, labels="LA", loc0="lb", loc1="lb", text_line_pad=1, ax=None):
    if ax is None:
        ax = plt.gca()

    dpi = ax.figure.dpi
    xlabel, ylabel = labels

    if loc0 == "lb" and loc1 == "lb":
        text_x = ax.add_text(
            0, 0, xlabel, pad=(2, 2), c="w", va="_", ha="l", transform=ax.transAxes
        )
        bbox = get_text_bbox(text_x)
        x = bbox.x1 + text_line_pad / 72 * dpi
        y = (bbox.y0 + bbox.y1) / 2 + text_line_pad / 72 * dpi
        x, y = ax.transData.inverted().transform((x, y))
        line = ax.plot(
            [x, x + length, x + length],
            [y, y, y - length],
            c="w",
            lw=0.5,
            solid_capstyle="butt",
        )
        text_y = ax.add_text(
            x + length,
            y - length,
            ylabel,
            va="_",
            ha="c",
            color="w",
            pad=(0, text_line_pad),
        )
        return {"text_x": text_x, "text_y": text_y, "line": line[0]}
    else:
        raise NotImplementedError


def add_ocular_condition_symbol(
    condition: str,
    x=0.5,
    y=1,
    r=5,
    sep=1,
    xpad=0,
    ypad=0,
    lw=2,
    scale=1,
    ax=None,
    va="bottom",
    ha="center",
    cross_color="k",
    bg_color=None,
    bg_scale=1.2,
    delense=False,
    lw_arrow=1,
    **kwargs,
):
    import numpy as np
    from matplotlib.patches import Ellipse, FancyArrowPatch

    va = va.lower().strip()[0]
    ha = ha.lower().strip()[0]

    if ax is None:
        ax = plt.gca()

    r, sep = r * scale, sep * scale

    delense = False

    if "B" in condition:
        condition = "B"
    else:
        condition = condition.strip().lower()

        if "e" in condition or "n" in condition:
            delense = True

        condition = condition[0].lower()
        condition = dict(s="b").get(condition, condition)

    color = {
        "l": "C0",
        "r": "C1",
        "b": "C2",
        "0": "C5",
        "B": "lightseagreen",
        "i": "darkblue",
        "c": "darkorchid",
        "n": "k",
    }[condition]

    pt2w = ax.figure.dpi / (72 * ax.bbox.width)
    pt2h = ax.figure.dpi / (72 * ax.bbox.height)
    width = r * 2 * pt2w
    height = r * 2 * pt2h

    kw = dict(
        dict(
            width=width,
            height=height,
            clip_on=False,
            transform=ax.transAxes,
            facecolor=color,
            linewidth=0,
        ),
        **kwargs,
    )

    if va == "b":
        y = y + height / 2
    elif va == "t":
        y = y - height / 2

    if ha == "l":
        x = x + width + sep * pt2w / 2
    elif ha == "r":
        x = x - width - sep * pt2w / 2

    d = (r + sep / 2) * pt2w
    x = x + xpad * pt2w
    y = y + ypad * pt2h
    xy_l = (x - d, y)
    xy_r = (x + d, y)

    artists = {}

    if bg_color is not None:
        bkw = kw.copy()
        bkw["width"] *= bg_scale
        bkw["height"] *= bg_scale
        bkw["facecolor"] = bg_color
        artists["bg_l"] = ax.add_patch(Ellipse(xy_l, **bkw))
        artists["bg_r"] = ax.add_patch(Ellipse(xy_r, **bkw))

    kw_l = kw.copy()
    kw_r = kw.copy()

    if delense:
        if condition == "l":
            kw_r["facecolor"] = "k"
        if condition == "r":
            kw_l["facecolor"] = "k"

    kw_ = dict(
        width=width / 3,
        height=height / 3,
        clip_on=False,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor="k",
        linewidth=0.5 * scale,
    )

    if condition in "Bic":
        xy_l2 = np.array(xy_r) - (0, height * 0.8)
        xy_r2 = np.array(xy_l) - (0, height * 0.8)

        a = width / 6
        b = height / 6
        cy_ = xy_l[1] - 0.85 * height
        cx_ = xy_r2[0] - a * np.sqrt(1 - (height * 0.05 / b) ** 2)
        ax.add_patch(Ellipse((cx_, cy_), **kw_))

        if condition in "Bi":
            artists["ipsi_arrow"] = ax.add_patch(
                FancyArrowPatch(
                    xy_l,
                    xy_l2,
                    shrinkA=0,
                    shrinkB=0,
                    arrowstyle="-|>",
                    mutation_scale=4 * scale,
                    clip_on=False,
                    joinstyle="miter",
                    color="k",
                    transform=ax.transAxes,
                    lw=lw_arrow * scale,
                )
            )
        if condition in "Bc":
            artists["contra_arrow"] = ax.add_patch(
                FancyArrowPatch(
                    xy_r,
                    xy_r2,
                    shrinkA=0,
                    shrinkB=0,
                    arrowstyle="-|>",
                    mutation_scale=4 * scale,
                    clip_on=False,
                    joinstyle="miter",
                    color="k",
                    transform=ax.transAxes,
                    lw=lw_arrow * scale,
                )
            )

        if condition in "Bi":
            artists["inter_arrow"] = ax.add_patch(
                FancyArrowPatch(
                    xy_l2 - (0, height * 0.1),
                    xy_r2 - (0, height * 0.1),
                    shrinkA=0,
                    shrinkB=0,
                    arrowstyle="-|>",
                    mutation_scale=4 * scale,
                    clip_on=False,
                    joinstyle="miter",
                    color="k",
                    lw=lw_arrow * scale,
                    connectionstyle="arc3,rad=-.5",
                    transform=ax.transAxes,
                )
            )

    artists["fg_l"] = pl = ax.add_patch(Ellipse(xy_l, **kw_l))
    artists["fg_r"] = pr = ax.add_patch(Ellipse(xy_r, **kw_r))

    for conditions, xy, p in zip(("l0", "r0"), (xy_r, xy_l), (pr, pl)):
        if condition in conditions and not delense:
            cx, cy = xy
            w = width / 2 / (2**0.5)
            h = height / 2 / (2**0.5)
            ax.plot(
                [cx - w, cx + w],
                [[cy - h, cy + h], [cy + h, cy - h]],
                transform=ax.transAxes,
                color=cross_color,
                lw=lw * scale,
                solid_capstyle="butt",
                clip_path=p,
            )

    return artists
