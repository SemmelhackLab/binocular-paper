import matplotlib.pyplot as plt


def get_asterisks(pval):
    if pval <= 0.001:
        return "***"
    elif pval <= 0.01:
        return "**"
    elif pval <= 0.05:
        return "*"
    else:
        return ""


def add_brackets(
    xl, xr, yt, yl, yr=None, text=None, pad=(0, 0), plot_kw=None, text_kw=None, ax=None
):
    if not text:
        return

    if ax is None:
        ax = plt.gca()
    if yr is None:
        yr = yl

    if plot_kw is None:
        plot_kw = {}

    plot_kw = dict(dict(color="k", clip_on=False), **plot_kw)

    if text_kw is None:
        text_kw = {}

    text_kw = dict(dict(annotation_clip=False), **text_kw)

    ax.plot([xl, xl, xr, xr], [yl, yt, yt, yr], **plot_kw)
    if text is not None:
        ax.annotate(
            text,
            ((xl + xr) / 2, yt),
            pad,
            textcoords="offset points",
            va="bottom",
            ha="center",
            **text_kw,
        )
