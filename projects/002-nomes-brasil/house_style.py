"""House style for the figures of DataVerso exploration 002.

The module holds the colours, canvas sizes and Matplotlib defaults of the figures, the
header and footer drawn around each of them, the axes styling and the number formatting
of the tick labels. Later projects copy it and extend it when they repeat something new.

This copy of exploration 001's module adds the colour of men's names, ``MEN``, and lets
``style_axes`` leave out the unit, for panels that share the unit of another.
The life-cycle palette and colour mixes are shared by the figures of exploration 002.
"""

from pathlib import Path

from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

# The IBM Plex fonts live at the repository root, two levels above this file.
FONT_DIR = Path(__file__).resolve().parents[2] / "fonts"
SIGNATURE = "github.com/jhlopesalves/DataVerso"  # bottom right of every figure

# --- Colours --------------------------------------------------------------------------
PAPER = "#fbfaf7"  # background
INK = "#1f1f1f"  # title, labels on the data and the zero baseline
SUBTLE = "#4a4a48"  # subtitle
MUTED = "#6b6b6b"  # units, tick labels and footer
AXIS = "#8f8b83"  # neutral spine, tick marks and reference lines
RULE = "#dcd8cf"  # gridlines and footer rule
CONTEXT = "#b4b0a8"  # de-emphasised marks
ACCENT = "#0f766e"  # header bar and the series that carries the argument
# Exploration 002 encodes men's names in MEN and women's names in ACCENT. MEN has a
# contrast of 4.8:1 with PAPER and stays apart from ACCENT under simulated deuteranopia
# and protanopia.
MEN = "#b45309"
BOTH = "#66625d"  # names used by both sexes
# Life-cycle groups 1 to 5 use colours distinct from the teal and ochre sex classes.
GROUP_COLOURS = ("#8a6a8c", "#b0717f", "#a88a45", "#6b82a8", "#7d9868")

# --- Canvases: (width, height) in inches, and dots per inch ---------------------------
SOCIAL_SIZE = (7.2, 9.0)
SOCIAL_DPI = 150
LANDSCAPE_SIZE = (9.0, 6.0)
LANDSCAPE_DPI = 200

# --- Type: sizes in points, line spacing as a multiple of the font size ---------------
TITLE_FONT = "IBM Plex Serif"
TEXT_FONT = "IBM Plex Sans"
TITLE_SIZE, TITLE_SPACING = 20.0, 1.15
SUBTITLE_SIZE, SUBTITLE_SPACING = 11.0, 1.4
NOTE_SIZE, NOTE_SPACING = 7.5, 1.45
UNIT_SIZE = 9.0
TICK_SIZE = 9.0
LABEL_SIZE = 9.5

# --- Lengths in points (1/72 inch) ----------------------------------------------------
MARGIN = 26.0  # outer page margin
ACCENT_BAR_WIDTH, ACCENT_BAR_HEIGHT = 34.0, 3.0
ACCENT_BAR_GAP = 12.0  # accent bar to title
TITLE_GAP = 7.0  # title to subtitle
HEADER_GAP = 18.0  # subtitle to plotting area
FOOTER_GAP = 18.0  # plotting area to footer rule
FOOTER_RULE_GAP = 7.0  # footer rule to footer text
FOOTER_BOTTOM = MARGIN - 4.0  # lower edge of the footer text, 4 pt into the margin
UNIT_OFFSET = 8.0  # top of the axes to the unit label
RULE_WIDTH = 0.8
GRID_WIDTH = 0.6
SPINE_WIDTH = 0.8
ZERO_WIDTH = 1.0  # bottom spine when the y axis starts at zero

FOOTER_LABELS = {
    "pt": {"source": "Fonte", "note": "Nota"},
    "en": {"source": "Source", "note": "Note"},
}

# Defaults for every figure, applied with plt.rcParams.update(RC_PARAMS).
RC_PARAMS: dict[str, object] = {
    "figure.dpi": SOCIAL_DPI,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.axisbelow": True,  # gridlines behind bars and lines
    "axes.labelcolor": MUTED,
    "axes.labelsize": LABEL_SIZE,
    "axes.labelpad": 7.0,
    "font.family": TEXT_FONT,
    "text.color": INK,
    "xtick.color": MUTED,  # tick marks of native axes, such as the map's colour key
    "xtick.labelcolor": MUTED,
    "xtick.labelsize": TICK_SIZE,
    "xtick.major.pad": 4.0,
    "ytick.labelcolor": MUTED,
    "ytick.labelsize": TICK_SIZE,
    "ytick.major.size": 0.0,  # gridlines replace the y tick marks
    "ytick.major.pad": 6.0,
    # Export at the figure's own dpi and full canvas, so each PNG keeps its pixel size.
    "savefig.dpi": "figure",
    "savefig.bbox": "standard",
    "pdf.fonttype": 42,  # embed TrueType fonts in PDF output
}


def register_fonts() -> None:
    """Register the IBM Plex fonts from the repository's ``fonts/`` directory."""
    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(path)


def tint(colour: str, amount: float) -> tuple[float, float, float]:
    """Mix a colour with PAPER. An amount of 1 keeps the colour, and 0 gives PAPER."""
    red, green, blue = to_rgb(colour)
    paper_red, paper_green, paper_blue = to_rgb(PAPER)
    return (
        amount * red + (1 - amount) * paper_red,
        amount * green + (1 - amount) * paper_green,
        amount * blue + (1 - amount) * paper_blue,
    )


def shade(colour: str, amount: float) -> tuple[float, float, float]:
    """Mix a colour with INK. An amount of 1 gives INK, and 0 keeps the colour."""
    red, green, blue = to_rgb(colour)
    ink_red, ink_green, ink_blue = to_rgb(INK)
    return (
        (1 - amount) * red + amount * ink_red,
        (1 - amount) * green + amount * ink_green,
        (1 - amount) * blue + amount * ink_blue,
    )


def frame(
    fig: Figure, *, title: str, subtitle: str, source: str, note: str, lang: str
) -> None:
    """Draw the header and footer of ``fig`` and fit the plotting area between them.

    The texts carry explicit line breaks, from which their heights are computed.

    Parameters
    ----------
    fig : Figure
        Figure to frame, once, before or after its axes are created.
    title, subtitle : str
        Header texts.
    source, note : str
        Footer texts, without the "Source:" and "Note:" labels.
    lang : {"pt", "en"}
        Language of the footer labels.
    """
    width, height = fig.get_size_inches() * 72  # canvas in points
    top = _add_header(fig, title, subtitle, width, height) - HEADER_GAP
    bottom = _add_footer(fig, source, note, lang, width, height) + FOOTER_GAP
    # Constrained layout measures each axes' margins from where the axes stand.
    # Starting the grid at the canvas edge lets the y tick labels, rather than the unit
    # label fixed at the page margin, decide where the axes settle.
    fig.subplots_adjust(left=0)
    fig.set_layout_engine(
        "constrained",
        rect=(
            MARGIN / width,
            bottom / height,
            1 - 2 * MARGIN / width,
            (top - bottom) / height,
        ),
        w_pad=0,
        h_pad=0,
        hspace=0.05,
    )


def _text_height(text: str, size: float, spacing: float) -> float:
    """Return the height in points of a text drawn with a numeric line spacing."""
    # Matplotlib gives each line spacing * (ascent + descent) of the font, which is
    # one em for IBM Plex, so the height follows from the number of lines.
    return (text.count("\n") + 1) * spacing * size


def _add_header(
    fig: Figure, title: str, subtitle: str, width: float, height: float
) -> float:
    """Draw the accent bar, title and subtitle.

    Return the header's lower edge in points.
    """
    left = MARGIN / width
    y = height - MARGIN  # running position, in points above the bottom edge
    bar = Rectangle(
        (left, (y - ACCENT_BAR_HEIGHT) / height),
        ACCENT_BAR_WIDTH / width,
        ACCENT_BAR_HEIGHT / height,
        transform=fig.transFigure,
        color=ACCENT,
        linewidth=0,
    )
    fig.add_artist(bar)
    y -= ACCENT_BAR_HEIGHT + ACCENT_BAR_GAP
    # parse_math=False keeps two "R$" in one text from being read as mathtext.
    fig.text(
        left,
        y / height,
        title,
        va="top",
        fontfamily=TITLE_FONT,
        fontsize=TITLE_SIZE,
        fontweight="semibold",
        color=INK,
        linespacing=TITLE_SPACING,
        parse_math=False,
    )
    y -= _text_height(title, TITLE_SIZE, TITLE_SPACING) + TITLE_GAP
    fig.text(
        left,
        y / height,
        subtitle,
        va="top",
        fontsize=SUBTITLE_SIZE,
        color=SUBTLE,
        linespacing=SUBTITLE_SPACING,
        parse_math=False,
    )
    return y - _text_height(subtitle, SUBTITLE_SIZE, SUBTITLE_SPACING)


def _add_footer(
    fig: Figure, source: str, note: str, lang: str, width: float, height: float
) -> float:
    """Draw the source, note, signature and rule; return the rule's height in points."""
    labels = FOOTER_LABELS[lang]
    text = f"{labels['source']}: {source}\n{labels['note']}: {note}"
    left = MARGIN / width
    top = FOOTER_BOTTOM + _text_height(text, NOTE_SIZE, NOTE_SPACING)
    style = dict(
        fontsize=NOTE_SIZE, color=MUTED, linespacing=NOTE_SPACING, parse_math=False
    )
    fig.text(left, FOOTER_BOTTOM / height, text, va="bottom", **style)
    # The signature sits at the right margin, level with the first footer line.
    right = 1 - left
    fig.text(
        right,
        top / height,
        SIGNATURE,
        ha="right",
        va="top",
        fontweight="semibold",
        **style,
    )
    rule = top + FOOTER_RULE_GAP
    line = Line2D(
        [left, right],
        [rule / height] * 2,
        transform=fig.transFigure,
        color=RULE,
        linewidth=RULE_WIDTH,
    )
    fig.add_artist(line)
    return rule


def style_axes(
    ax: Axes,
    unit: str | None,
    *,
    xlabel: str | None = None,
    zero_baseline: bool = False,
) -> None:
    """Apply the house axes style to ``ax`` and write ``unit`` above it.

    Call it once the data are drawn, so that the zero baseline takes them into account.

    Parameters
    ----------
    ax : Axes
        Axes to style.
    unit : str or None
        Unit of the y axis, written above the axes at the left page margin. Use None
        when the unit is written once for several panels.
    xlabel : str, optional
        Label of the x axis.
    zero_baseline : bool, default False
        Start the y axis at zero unless an observation lies below it, and draw the
        bottom spine dark whenever the axis starts at zero.
    """
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", color=RULE, linewidth=GRID_WIDTH)
    # Tick sizes, pads and label colours come from RC_PARAMS. The marks take the axis
    # colour.
    ax.tick_params(axis="x", color=AXIS)

    # The figure fraction puts the unit at the page margin, in line with the header
    # and footer. The axes fraction keeps the unit just above the axes.
    if unit is not None:
        page_width = ax.get_figure(root=True).get_figwidth() * 72
        ax.annotate(
            unit,
            (MARGIN / page_width, 1),
            xycoords=("figure fraction", "axes fraction"),
            xytext=(0, UNIT_OFFSET),
            textcoords="offset points",
            va="bottom",
            fontsize=UNIT_SIZE,
            color=MUTED,
        )
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if zero_baseline and ax.dataLim.y0 >= 0:
        ax.set_ylim(bottom=0)
    at_zero = zero_baseline and ax.get_ylim()[0] == 0
    ax.spines["bottom"].set_color(INK if at_zero else AXIS)
    ax.spines["bottom"].set_linewidth(ZERO_WIDTH if at_zero else SPINE_WIDTH)


def _format_number(value: float, lang: str, decimals: int) -> str:
    """Write ``value`` with the thousands and decimal separators of ``lang``."""
    # round() takes halves to the even neighbour. Reset a zero result to 0.0 so that a
    # small negative value does not print as "-0".
    rounded = round(float(value), decimals)
    if rounded == 0:
        rounded = 0.0
    text = f"{rounded:,.{decimals}f}"
    # Portuguese swaps the English separators: 1,234.5 becomes 1.234,5.
    return text.translate(str.maketrans(",.", ".,")) if lang == "pt" else text


def format_brl(value: float, lang: str, decimals: int = 0) -> str:
    """Write an amount in Brazilian reais.

    The minus sign precedes ``R$``, Portuguese puts a space after it, and halves round
    to the even neighbour.

    Examples
    --------
    >>> format_brl(1234567.891, "pt")
    'R$ 1.234.568'
    >>> format_brl(1234.5, "en", decimals=2)
    'R$1,234.50'
    >>> format_brl(-2500, "pt")
    '-R$ 2.500'
    >>> format_brl(2.5, "en")
    'R$2'
    """
    text = _format_number(value, lang, decimals)
    sign, digits = ("-", text[1:]) if text.startswith("-") else ("", text)
    space = " " if lang == "pt" else ""
    return f"{sign}R${space}{digits}"


def number_formatter(lang: str, decimals: int = 0) -> FuncFormatter:
    """Return a tick formatter that writes numbers with the separators of ``lang``."""
    return FuncFormatter(lambda value, _position: _format_number(value, lang, decimals))
