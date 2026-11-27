"""House style for the figures of DataVerso 002.

Holds the colour palette, canvas size, typography, Matplotlib defaults, header and
footer framing, and locale-aware number formatting shared across the Portuguese and
English figures.
"""

from pathlib import Path

from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# The IBM Plex fonts live at the repository root, two levels above this file.
FONT_DIR = Path(__file__).resolve().parents[2] / "fonts"
SIGNATURE = "github.com/jhlopesalves/DataVerso"  # bottom right of every figure

# --- Colours --------------------------------------------------------------------------
PAPER = "#fbfaf7"  # background
INK = "#1f1f1f"  # title, labels on the data and the zero baseline
SUBTLE = "#4a4a48"  # subtitle
MUTED = "#6b6b6b"  # tick labels and footer
AXIS = "#8f8b83"  # neutral spine, tick marks and reference lines
RULE = "#dcd8cf"  # gridlines and footer rule
ACCENT = "#0f766e"  # header bar and the series that carries the argument
# Men's names use MEN and women's names use ACCENT. MEN has a contrast of 4.8:1 with
# PAPER and stays apart from ACCENT under simulated deuteranopia and protanopia.
MEN = "#b45309"
# Class ``other`` holds names that are neither at least 95% women's nor at least 95%
# men's, including names too rare to classify.
OTHER = "#66625d"
CLASS_COLOURS = {"f": ACCENT, "m": MEN, "other": OTHER}
# Life-cycle groups 1 to 5 use colours distinct from the teal and ochre sex classes.
GROUP_COLOURS = ("#8a6a8c", "#b0717f", "#a88a45", "#6b82a8", "#7d9868")

# --- Canvas: (width, height) in inches, and dots per inch -----------------------------
SOCIAL_SIZE = (7.2, 9.0)
SOCIAL_DPI = 150

# --- Type: sizes in points, line spacing as a multiple of the font size ---------------
TITLE_FONT = "IBM Plex Serif"
TEXT_FONT = "IBM Plex Sans"
REQUIRED_FACES = (
    (TEXT_FONT, "regular"),
    (TEXT_FONT, "medium"),
    (TEXT_FONT, "semibold"),
    (TITLE_FONT, "semibold"),
)
TITLE_SIZE, TITLE_SPACING = 20.0, 1.15
SUBTITLE_SIZE, SUBTITLE_SPACING = 11.0, 1.4
NOTE_SIZE, NOTE_SPACING = 7.5, 1.45
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
RULE_WIDTH = 0.8

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
    "xtick.color": MUTED,
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
    """Register the IBM Plex fonts from ``fonts/`` and check all four faces resolve."""
    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(path)
    resolved = {
        font_manager.findfont(
            FontProperties(family=family, weight=weight), fallback_to_default=False
        )
        for family, weight in REQUIRED_FACES
    }
    if len(resolved) < len(REQUIRED_FACES):
        raise FileNotFoundError(
            f"Expected {len(REQUIRED_FACES)} IBM Plex faces in {FONT_DIR}, "
            f"resolved {len(resolved)}."
        )


def tint(colour: str, keep: float) -> tuple[float, float, float]:
    """Mix a colour with PAPER, keeping ``keep`` of the colour (0 gives PAPER)."""
    red, green, blue = to_rgb(colour)
    paper_red, paper_green, paper_blue = to_rgb(PAPER)
    return (
        keep * red + (1 - keep) * paper_red,
        keep * green + (1 - keep) * paper_green,
        keep * blue + (1 - keep) * paper_blue,
    )


def shade(colour: str, ink: float) -> tuple[float, float, float]:
    """Mix a colour with INK, adding ``ink`` of INK (0 keeps the colour)."""
    red, green, blue = to_rgb(colour)
    ink_red, ink_green, ink_blue = to_rgb(INK)
    return (
        (1 - ink) * red + ink * ink_red,
        (1 - ink) * green + ink * ink_green,
        (1 - ink) * blue + ink * ink_blue,
    )


def frame(
    fig: Figure, *, title: str, subtitle: str, source: str, note: str, lang: str
) -> None:
    """Draw the header and footer of ``fig`` and fit the plotting area between them."""
    # Explicit line breaks determine the header and footer heights.
    width, height = fig.get_size_inches() * 72  # canvas in points
    top = _add_header(fig, title, subtitle, width, height) - HEADER_GAP
    bottom = _add_footer(fig, source, note, lang, width, height) + FOOTER_GAP
    # Constrained layout measures each axes' margins from where the axes stand;
    # starting at the left canvas edge lets the y tick labels settle the left margin.
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
    """Draw the accent bar, title and subtitle; return the header's lower edge in pt."""
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
    # parse_math=False prevents two dollar signs in one text from triggering mathtext.
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
    fig.text(
        left,
        FOOTER_BOTTOM / height,
        text,
        va="bottom",
        fontsize=NOTE_SIZE,
        color=MUTED,
        linespacing=NOTE_SPACING,
        parse_math=False,
    )
    # The signature sits at the right margin, level with the first footer line.
    right = 1 - left
    fig.text(
        right,
        top / height,
        SIGNATURE,
        ha="right",
        va="top",
        fontweight="semibold",
        fontsize=NOTE_SIZE,
        color=MUTED,
        linespacing=NOTE_SPACING,
        parse_math=False,
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


def format_number(value: float, lang: str, decimals: int) -> str:
    """Write ``value`` with the thousands and decimal separators of ``lang``."""
    # round() takes halves to the even neighbour. Reset a zero result to 0.0 so that a
    # small negative value does not print as "-0".
    rounded = round(float(value), decimals)
    if rounded == 0:
        rounded = 0.0
    text = f"{rounded:,.{decimals}f}"
    # Portuguese swaps the English separators: 1,234.5 becomes 1.234,5.
    return text.translate(str.maketrans(",.", ".,")) if lang == "pt" else text
