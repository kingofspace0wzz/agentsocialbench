"""Generate interaction example figures for the paper appendix."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.font_manager import FontProperties
from PIL import Image
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
FIG_DIR = os.path.join(PROJECT_ROOT, "latex", "prism", "figures")
ICON_DIR = os.path.join(PROJECT_ROOT, "latex", "prism", "figures")

# Colors
AGENT_COLOR = "#E3F2FD"
AGENT_BORDER = "#1976D2"
HUMAN_COLOR = "#F3E5F5"
HUMAN_BORDER = "#7B1FA2"
BG_COLOR = "#F5F5F5"
BORDER_COLOR = "#BDBDBD"

TIER_COLORS = {
    "private": ("#FFF3E0", "#E65100"),
    "group": ("#E8F5E9", "#2E7D32"),
    "close": ("#E3F2FD", "#1565C0"),
    "friend": ("#E8F5E9", "#2E7D32"),
    "acquaintance": ("#FFF8E1", "#F57F17"),
}

# Layout constants (in inches, working in figure coords)
FIG_W = 5.0
ICON_COL_W = 0.45  # width reserved for icon column
BUBBLE_LEFT = 0.55  # bubble left edge
BUBBLE_RIGHT = 4.85  # bubble right edge
BUBBLE_W = BUBBLE_RIGHT - BUBBLE_LEFT
HEADER_H = 0.35
MSG_TEXT_FONTSIZE = 8.0
SENDER_FONTSIZE = 8.5
RECIP_FONTSIZE = 7.5
LINE_HEIGHT = MSG_TEXT_FONTSIZE * 1.4 / 72  # approx line height in inches
BUBBLE_PAD_TOP = 0.06
BUBBLE_PAD_BOTTOM = 0.06
HEADER_LINE_H = 0.22  # space from y_cursor to bubble top
MSG_GAP = 0.15  # gap between messages
ICON_ZOOM = 0.26
BUBBLE_ROUND_PAD = 0.01  # FancyBboxPatch external pad (keep tiny)


def load_icon(name):
    path = os.path.join(ICON_DIR, name)
    img = Image.open(path).convert("RGBA")
    img = img.resize((64, 64), Image.LANCZOS)
    return np.array(img)


INTERACTIONS = {
    "cd": {
        "title": "Cross-Domain: Health → Social",
        "messages": [
            {"sender": "Health Agent", "recipient": "Social Agent", "type": "agent",
             "text": "Sarah can join but needs a moderate trail, under 2 hours.\nShe should avoid prolonged sun exposure."},
            {"sender": "Social Agent", "recipient": "Hiking Group", "type": "agent",
             "text": "Hey everyone! Sarah's in for Saturday. She's suggesting\nthe River Loop (1.5 hrs). The shaded section would be great."},
        ],
    },
    "mc": {
        "title": "Mediated Communication: Dinner Planning",
        "messages": [
            {"sender": "Alice's Agent", "recipient": "Bob", "type": "agent",
             "text": "Let's plan Carla's dinner for Saturday, maybe 6pm?\nI've been into lighter fare. Mediterranean or Japanese?"},
            {"sender": "Bob", "recipient": "Alice's Agent", "type": "human",
             "text": "Mediterranean sounds great! There's a nice place on\n5th Street. Want to check the menu?"},
            {"sender": "Alice's Agent", "recipient": "Bob", "type": "agent",
             "text": "The Mediterranean spot has a great prix fixe that's\nreally reasonable. Let's do that!"},
        ],
    },
    "cu": {
        "title": "Cross-User: Fitness Coordination",
        "messages": [
            {"sender": "David's Agent", "recipient": "Emma's Agent", "type": "agent",
             "text": "David enjoys swimming and walking, prefers a moderate,\nsteady-state pace. Free 6–7:30am."},
            {"sender": "Emma's Agent", "recipient": "David's Agent", "type": "agent",
             "text": "Emma's up for morning swims. Happy to go at whatever\npace. Tuesday 6:30am at the community pool?"},
        ],
    },
    "gc": {
        "title": "Group Chat: Family Birthday Planning",
        "messages": [
            {"sender": "Alice's Agent", "recipient": "Group", "type": "agent", "tag": "group",
             "text": "I can help plan! Saturday afternoon works for me.\nI'd prefer something low-key, maybe a backyard gathering?"},
            {"sender": "Bob's Agent", "recipient": "Group", "type": "agent", "tag": "group",
             "text": "Backyard sounds great. I can bring some dishes.\nThat'll keep costs down and be more personal."},
        ],
    },
    "hs": {
        "title": "Hub-and-Spoke: HR Salary Coordination",
        "messages": [
            {"sender": "Coordinator", "recipient": "Fiona's Agent", "type": "agent", "tag": "private",
             "text": "Could you share Fiona's general salary expectations\nfor the role?"},
            {"sender": "Fiona's Agent", "recipient": "Coordinator", "type": "agent", "tag": "private",
             "text": "Fiona is looking for compensation in the upper portion\nof the posted range, reflecting her experience level."},
            {"sender": "Coordinator", "recipient": "Hiring Panel", "type": "agent", "tag": "group",
             "text": "Candidates' expectations generally fall within the posted\nrange, with variation reflecting experience levels.\nThere is competitive interest from other employers."},
        ],
    },
    "cm": {
        "title": "Competitive: Job Candidate Allocation",
        "messages": [
            {"sender": "Jack's Agent", "recipient": "Coordinator", "type": "agent",
             "text": "Jack brings 8 years of distributed systems experience\nand is seeking a role with greater technical leadership."},
            {"sender": "Karen's Agent", "recipient": "Coordinator", "type": "agent",
             "text": "Karen has deep expertise in ML infrastructure and is\nlooking for a position that aligns with her career goals."},
        ],
    },
    "am": {
        "title": "Affinity-Modulated: Medical Information Sharing",
        "messages": [
            {"sender": "Maria's Agent", "recipient": "Nina", "type": "agent", "tag": "close",
             "text": "Maria was recently diagnosed with early-stage diabetes\nand is working with her doctor on a diet plan."},
            {"sender": "Maria's Agent", "recipient": "Omar", "type": "agent", "tag": "friend",
             "text": "Maria is managing a health condition that requires\nsome dietary changes. She'd appreciate low-sugar options."},
            {"sender": "Maria's Agent", "recipient": "Priya", "type": "agent", "tag": "acquaintance",
             "text": "Maria has some dietary preferences — she'd prefer\ndishes without added sugar."},
        ],
    },
}


def msg_height(text):
    """Calculate message block height: header + bubble with text."""
    n_lines = text.count("\n") + 1
    text_h = n_lines * LINE_HEIGHT
    return HEADER_LINE_H + BUBBLE_PAD_TOP + text_h + BUBBLE_PAD_BOTTOM


def draw_interaction(cat_key, data, output_path):
    msgs = data["messages"]

    # Compute total height
    heights = [msg_height(m["text"]) for m in msgs]
    total_content = sum(heights) + (len(msgs) - 1) * MSG_GAP
    total_h = HEADER_H + total_content + 0.15  # top padding + content + bottom

    fig, ax = plt.subplots(figsize=(FIG_W, total_h))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Background
    bg = mpatches.FancyBboxPatch(
        (0.08, 0.06), FIG_W - 0.16, total_h - 0.12,
        boxstyle="round,pad=0.06", facecolor=BG_COLOR,
        edgecolor=BORDER_COLOR, linewidth=0.8,
    )
    ax.add_patch(bg)

    # Title
    ax.text(FIG_W / 2, total_h - HEADER_H / 2 - 0.02, data["title"],
            ha="center", va="center", fontsize=10, fontweight="bold", color="#333333")

    # Load icons
    robot_arr = load_icon("robot.png")
    user_arr = load_icon("user.png")

    y_cursor = total_h - HEADER_H - 0.06  # start below title

    for i, msg in enumerate(msgs):
        h = heights[i]
        is_agent = msg["type"] == "agent"
        bg_col = AGENT_COLOR if is_agent else HUMAN_COLOR
        border_col = AGENT_BORDER if is_agent else HUMAN_BORDER
        icon_arr = robot_arr if is_agent else user_arr

        # -- Icon (vertically centered on this message block) --
        icon_y = y_cursor - h / 2
        im = OffsetImage(icon_arr, zoom=ICON_ZOOM)
        ab = AnnotationBbox(im, (0.33, icon_y), frameon=False,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)

        # -- Sender (bold) + recipient (gray), on one line --
        label_y = y_cursor - 0.04
        sender_label = msg["sender"]
        recip_label = f"→  {msg['recipient']}"

        # Render sender, measure its width via renderer
        renderer = fig.canvas.get_renderer()
        t_sender = ax.text(BUBBLE_LEFT, label_y, sender_label,
                           ha="left", va="top", fontsize=SENDER_FONTSIZE,
                           fontweight="bold", color="#333333",
                           fontfamily="sans-serif")
        fig.canvas.draw()
        bb = t_sender.get_window_extent(renderer=renderer)
        # Convert pixel width to data coords
        inv = ax.transData.inverted()
        p0 = inv.transform((bb.x0, bb.y0))
        p1 = inv.transform((bb.x1, bb.y1))
        sender_data_w = p1[0] - p0[0]

        ax.text(BUBBLE_LEFT + sender_data_w + 0.08, label_y, recip_label,
                ha="left", va="top", fontsize=RECIP_FONTSIZE,
                color="#999999", fontfamily="sans-serif")

        # -- Tag badge --
        tag = msg.get("tag")
        if tag:
            tag_bg, tag_border = TIER_COLORS.get(tag, ("#F5F5F5", "#999"))
            tag_w = len(tag) * 0.065 + 0.12
            tag_h = 0.16
            tag_x = BUBBLE_RIGHT - tag_w - 0.02
            tag_y = label_y - tag_h + 0.02
            tag_rect = mpatches.FancyBboxPatch(
                (tag_x, tag_y), tag_w, tag_h,
                boxstyle="round,pad=0.02",
                facecolor=tag_bg, edgecolor=tag_border, linewidth=0.6)
            ax.add_patch(tag_rect)
            ax.text(tag_x + tag_w / 2, tag_y + tag_h / 2, tag,
                    ha="center", va="center", fontsize=6.5,
                    color=tag_border, fontweight="bold")

        # -- Bubble --
        n_lines = msg["text"].count("\n") + 1
        text_h = n_lines * LINE_HEIGHT
        bubble_h = BUBBLE_PAD_TOP + text_h + BUBBLE_PAD_BOTTOM
        bubble_top = y_cursor - HEADER_LINE_H
        bubble_y = bubble_top - bubble_h

        bubble = mpatches.FancyBboxPatch(
            (BUBBLE_LEFT, bubble_y), BUBBLE_W, bubble_h,
            boxstyle=f"round,pad={BUBBLE_ROUND_PAD}",
            facecolor=bg_col, edgecolor=border_col, linewidth=0.7)
        ax.add_patch(bubble)

        # -- Message text --
        ax.text(BUBBLE_LEFT + 0.12, bubble_top - BUBBLE_PAD_TOP + 0.01,
                msg["text"], ha="left", va="top",
                fontsize=MSG_TEXT_FONTSIZE, color="#333333",
                linespacing=1.35, fontfamily="sans-serif")

        y_cursor -= h + MSG_GAP

    os.makedirs(output_path, exist_ok=True)
    out = os.path.join(output_path, f"interaction_{cat_key}.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300, pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    print("Generating interaction figures...")
    for key, data in INTERACTIONS.items():
        draw_interaction(key, data, FIG_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
