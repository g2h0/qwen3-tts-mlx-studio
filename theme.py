import gradio as gr

COLORS = {
    "bg_primary": "#1a1b26",
    "bg_secondary": "#24283b",
    "bg_tertiary": "#414868",
    "fg_primary": "#c0caf5",
    "fg_secondary": "#a9b1d6",
    "fg_muted": "#565f89",
    "accent_blue": "#7aa2f7",
    "accent_cyan": "#7dcfff",
    "accent_green": "#9ece6a",
    "accent_magenta": "#bb9af7",
    "accent_orange": "#ff9e64",
    "accent_red": "#f7768e",
    "border": "#3b4261",
}

custom_css = """
/* Global background */
.gradio-container { background-color: #1a1b26 !important; color: #c0caf5 !important; }

/* Tab styling */
.tab-nav button { color: #a9b1d6 !important; font-size: 0.88em; padding: 8px 14px !important; }
.tab-nav button.selected { color: #7aa2f7 !important; border-bottom-color: #7aa2f7 !important; }

/* Primary button */
.primary { background-color: #7aa2f7 !important; color: #1a1b26 !important; font-weight: 600 !important; }

/* Audio player */
.audio-player { background-color: #24283b !important; }

/* Text inputs */
textarea, input[type="text"] { background-color: #24283b !important; color: #c0caf5 !important; border-color: #3b4261 !important; }

/* Global status bar at very bottom */
.status-bar { background-color: #16161e !important; border-top: 1px solid #3b4261 !important; }
.status-bar textarea { background: transparent !important; border: none !important; padding: 8px 14px !important; min-height: 58px !important; max-height: 58px !important; resize: none !important; overflow-y: auto !important; color: #a9b1d6 !important; font-size: 0.84em !important; line-height: 1.5 !important; }

/* Header */
.app-header { text-align: center; padding: 12px 16px 8px; }
.app-header h1 { color: #7aa2f7; font-size: 1.4em; margin: 0 0 2px; }
.app-header .subtitle { color: #565f89; font-size: 0.85em; margin: 0; }

/* Batch mode accordion */
.batch-accordion { border: 1px solid #3b4261 !important; border-radius: 8px; margin-top: 10px; }
.batch-accordion > .label-wrap { padding: 6px 12px !important; font-size: 0.88em !important; color: #565f89 !important; }

/* Script editor */
.script-editor textarea { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace !important; font-size: 0.88em; line-height: 1.6; }

/* Speaker assignment slots — color-coded borders */
.speaker-slot-0 { border-left: 3px solid #7aa2f7 !important; padding-left: 8px; }
.speaker-slot-1 { border-left: 3px solid #bb9af7 !important; padding-left: 8px; }
.speaker-slot-2 { border-left: 3px solid #9ece6a !important; padding-left: 8px; }
.speaker-slot-3 { border-left: 3px solid #ff9e64 !important; padding-left: 8px; }
.speaker-slot-4 { border-left: 3px solid #7dcfff !important; padding-left: 8px; }
.speaker-slot-5 { border-left: 3px solid #f7768e !important; padding-left: 8px; }
.speaker-slot-6 { border-left: 3px solid #e0af68 !important; padding-left: 8px; }
.speaker-slot-7 { border-left: 3px solid #73daca !important; padding-left: 8px; }

/* History table */
.history-table { font-size: 0.85em; }

/* Progress indicators */
.batch-progress { color: #9ece6a; font-weight: bold; }

/* YT Voice Clone step headers */
.yt-step p {
    color: #7aa2f7 !important;
    font-weight: 700 !important;
    font-size: 0.82em !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin: 10px 0 4px !important;
    padding: 4px 10px !important;
    border-left: 3px solid #7aa2f7 !important;
    background: rgba(122, 162, 247, 0.07) !important;
    border-radius: 0 4px 4px 0 !important;
}

/* Info notice box (voice cloning, YT clone) */
.info-notice {
    background: rgba(122, 162, 247, 0.07) !important;
    border: 1px solid rgba(122, 162, 247, 0.25) !important;
    border-left: 3px solid #7aa2f7 !important;
    border-radius: 0 6px 6px 0 !important;
    padding: 8px 14px !important;
    font-size: 0.84em !important;
    color: #a9b1d6 !important;
    margin-bottom: 6px !important;
    line-height: 1.5 !important;
}
.info-notice p { margin: 0 !important; }

/* Compact save-status field (label hidden) */
.save-status-text textarea {
    font-size: 0.8em !important;
    min-height: 32px !important;
    padding: 5px 8px !important;
    color: #565f89 !important;
    resize: none !important;
}
.save-status-text .label-wrap { display: none !important; }

/* Model status field in Settings — label visible, compact */
.model-status textarea {
    font-size: 0.82em !important;
    min-height: 32px !important;
    padding: 5px 8px !important;
    color: #a9b1d6 !important;
    resize: none !important;
}

/* Output column left border separator */
.output-col { border-left: 1px solid #3b4261; padding-left: 4px !important; }

/* Library save section accordion */
.lib-save-accordion { border: 1px dashed #3b4261 !important; border-radius: 6px !important; margin-top: 6px !important; }
.lib-save-accordion > .label-wrap { font-size: 0.88em !important; color: #565f89 !important; padding: 5px 10px !important; }

/* Settings panel groups */
.settings-group {
    background: rgba(36, 40, 59, 0.5) !important;
    border: 1px solid #3b4261 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

/* Section markdown headings inside tabs */
.tab-content h3 { font-size: 0.9em !important; color: #7aa2f7 !important; margin: 8px 0 4px !important; font-weight: 600 !important; }

/* Text length hint below text boxes */
.text-hint p { color: #565f89 !important; font-size: 0.8em !important; margin: 2px 0 4px !important; }
"""


def build_theme():
    """Build a custom Gradio theme."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e8ecfd", c100="#d1d9fb", c200="#a3b3f7",
            c300="#7aa2f7", c400="#5b8af5", c500="#7aa2f7",
            c600="#6690e6", c700="#5278d4", c800="#3e60c3", c900="#2a48b1",
            c950="#1a1b26",
        ),
        neutral_hue=gr.themes.Color(
            c50="#c0caf5", c100="#a9b1d6", c200="#9aa5ce",
            c300="#7982a9", c400="#565f89", c500="#414868",
            c600="#3b4261", c700="#24283b", c800="#1a1b26", c900="#16161e",
            c950="#0f0f14",
        ),
    ).set(
        body_background_fill="#1a1b26",
        body_text_color="#c0caf5",
        block_background_fill="#24283b",
        block_border_color="#3b4261",
        input_background_fill="#24283b",
        input_border_color="#3b4261",
        button_primary_background_fill="#7aa2f7",
        button_primary_text_color="#1a1b26",
    )
