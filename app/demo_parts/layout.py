DEMO_CSS = """
html, body, gradio-app, .gradio-container {
    width: 100%;
    height: 100vh;
    height: 100dvh;
    margin: 0;
    overflow: hidden !important;
}
.gradio-container { box-sizing: border-box; max-width: none !important; padding: 20px 32px 36px !important; }
.gradio-container footer { display: none !important; }
#demo-shell {
    height: calc(100vh - 80px);
    height: calc(100dvh - 80px);
    min-height: 0;
    gap: 12px;
    overflow: hidden;
}
#demo-header { flex: 0 0 auto; align-items: center; gap: 24px; flex-wrap: nowrap; overflow: hidden; }
#demo-header, #demo-header * { scrollbar-width: none !important; }
#demo-header::-webkit-scrollbar, #demo-header *::-webkit-scrollbar { display: none !important; }
#demo-title { flex: 1 1 auto; min-width: 0; }
#demo-title h1 { font-size: 24px; line-height: 1.2; margin: 0; }
#demo-header > .form { flex: 0 1 400px !important; min-width: 240px !important; }
#demo-key { min-width: 0 !important; }
#demo-tabs { display: flex; flex-direction: column; flex: 1; min-height: 0; overflow: hidden; }
#demo-tabs > .tab-nav, #demo-tabs > [role="tablist"] { flex: 0 0 auto; }
.workspace-tab { flex: 1; min-height: 0; padding: 12px 0 0 !important; overflow: hidden; }
.workspace-tab > div { height: 100%; min-height: 0; }
.workspace-row { height: 100%; min-height: 0; flex-wrap: nowrap !important; align-items: stretch; }
.workspace-panel {
    min-height: 0;
    min-width: 0 !important;
    height: 100%;
    padding: 16px;
    gap: 12px;
    overflow: hidden;
    background: var(--block-background-fill);
    border: 1px solid var(--block-border-color);
    border-radius: 12px;
}
* { scrollbar-width: thin; scrollbar-color: #667085 transparent; }
@supports selector(::-webkit-scrollbar) {
    * { scrollbar-width: auto; }
}
*::-webkit-scrollbar { width: 6px; height: 6px; }
*::-webkit-scrollbar-button { display: none; width: 0; height: 0; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb { background: #667085; border: 1px solid transparent; border-radius: 4px; background-clip: padding-box; }
.workspace-panel > :not(.ticket-input):not(.table-panel):not(.json-view):not(.single-output) { flex-shrink: 0; }
.workspace-panel > .form:has(.ticket-input) { display: flex; flex-direction: column; flex: 1 !important; min-height: 0; overflow: hidden; }
.workspace-panel > .form:has(.ticket-input) > :not(.ticket-input) { flex-shrink: 0; }
.ticket-input { display: flex; flex-direction: column; flex: 1 !important; min-height: 0 !important; overflow: hidden; }
.ticket-input label, .ticket-input .input-container { display: flex; flex-direction: column; flex: 1; min-height: 0; }
.ticket-input textarea { flex: 1; height: 100% !important; min-height: 80px !important; max-height: none !important; overflow: auto; resize: none; }
.single-output { flex: 1 !important; min-height: 0; overflow: auto; }
.table-panel { flex: 1 !important; height: 0; min-height: 0 !important; overflow: auto; }
.json-view { display: flex; flex-direction: column; flex: 1 !important; min-height: 0; overflow: hidden; }
.json-view > div { min-height: 0; }
.json-view .cm-editor { flex: 1; height: 100% !important; max-height: 100% !important; min-height: 0; }
.json-view .cm-scroller { min-height: 0 !important; overflow: auto !important; }
.json-view .wrap, .json-view .codemirror-wrapper { display: flex; flex-direction: column; flex: 1; min-height: 0; overflow: hidden; }
.problems-panel { max-height: 110px; overflow: auto; font-size: 13px; }
.batch-status { min-height: 24px; font-size: 13px; }
.summary-panel { font-size: 13px; }
#csv-upload { min-height: 176px !important; max-height: none; overflow: visible; }
@media (max-width: 899px) {
    html, body, gradio-app, .gradio-container { height: auto; min-height: 100vh; overflow: auto !important; }
    .gradio-container { padding: 12px 12px 24px !important; }
    #demo-shell { height: auto; overflow: visible; }
    #demo-header { flex-wrap: wrap; gap: 12px; }
    #demo-title { flex-basis: 100%; }
    #demo-title h1 { font-size: 22px; }
    #demo-header > .form { flex: 1 1 220px !important; min-width: 0 !important; }
    #demo-tabs, .workspace-tab { overflow: visible; }
    .workspace-row { flex-wrap: wrap !important; height: auto; }
    .workspace-panel { flex-basis: 100% !important; height: auto; }
    .ticket-input { min-height: 200px !important; }
    .table-panel { flex: none !important; height: 320px; }
    .json-view { flex: none !important; height: 360px; }
    .single-output { max-height: 480px; }
}
"""

TABLE_CSS = """
font-family: system-ui, sans-serif;
font-size: 13px;
color: var(--body-text-color, #101828);
height: 100%;
min-height: 0;
overflow: auto;
.table-scroll { height: 100%; min-height: 0; overflow: auto; }
* { scrollbar-width: thin; scrollbar-color: #667085 transparent; }
@supports selector(::-webkit-scrollbar) {
    * { scrollbar-width: auto; }
}
*::-webkit-scrollbar { width: 6px; height: 6px; }
*::-webkit-scrollbar-button { display: none; width: 0; height: 0; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb { background: #667085; border-radius: 4px; }
table { border-collapse: collapse; min-width: 100%; width: max-content; }
th, td { padding: 10px 12px; text-align: left; max-width: 80ch; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-bottom: 1px solid var(--block-border-color, #d0d5dd); }
th { position: sticky; top: 0; max-width: none; overflow: visible; text-overflow: clip; background: var(--block-background-fill, #fff); font-weight: 600; }
.urgent { color: #b42318; background: #fef3f2; border-radius: 5px; padding: 3px 7px; font-size: 11px; font-weight: 700; }
.empty { color: var(--body-text-color-subdued, #475467); padding: 24px 12px; }
"""

SUMMARY_CSS = """
font-family: system-ui, sans-serif;
color: var(--body-text-color, #101828);
font-size: 13px;
.summary-counts { display: flex; flex-wrap: wrap; gap: 8px 20px; }
.summary-categories { margin-top: 6px; line-height: 1.5; overflow-wrap: anywhere; }
"""

PROBLEMS_CSS = """
font-family: system-ui, sans-serif;
font-size: 13px;
color: var(--body-text-color, #101828);
ul { padding-left: 20px; margin: 6px 0; }
li { margin: 3px 0; overflow-wrap: anywhere; }
"""
