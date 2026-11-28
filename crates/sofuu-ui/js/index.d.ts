/**
 * `sofuu-ui` — declarative scene, widgets and effects surface.
 *
 * Generated from `sofuu.d.ts` in the host while the UI declarations still
 * live there (PLAN-SPLIT-UI-EFFECTS-EXTERNAL-REPO §2.1). The host owns the
 * global `sofuu.ui` / `sofuu.fx` types; this module is what a standalone
 * consumer imports.
 *
 * Do not edit by hand: `bash scripts/sync_ui_dts.sh`.
 */

// ---8<--- generated from sofuu.d.ts (do not edit below) ---

/* ==================================================================
 * Native 2D GPU UI (`sofuu.ui`, enabled with `--ui`)
 * ================================================================== */

export type UiThemeMode = 'light' | 'dark';
export type UiTone = 'neutral' | 'info' | 'ok' | 'warn' | 'danger';
export type UiKeyboardHint =
  | 'text' | 'email' | 'number' | 'phone' | 'url' | 'search' | 'password' | 'decimal';
export type UiStackAlign =
  | 'top_left' | 'top' | 'top_right'
  | 'left' | 'center' | 'right'
  | 'bottom_left' | 'bottom' | 'bottom_right';
export type UiImageFit = 'cover' | 'contain' | 'fill' | 'scale_down';
export type UiDragAxis = 'both' | 'horizontal' | 'vertical';

/** Optional edge pinning inside a `Stack`/`Overlay`. */
export interface UiInsetHint {
  left?: number;
  right?: number;
  top?: number;
  bottom?: number;
}

/** Opt-in per-node layout constraints (`Node.layout`). */
export interface UiLayoutHints {
  max_width?: number;
  max_height?: number;
  aspect_ratio?: number;
  flex_shrink?: number;
  align_self?: UiAlign;
  inset?: UiInsetHint;
  z?: number;
}

/** One inline styled run in a `RichText` node. */
export interface UiTextSpan {
  text: string;
  color?: number | null;
  weight?: number;
  mono?: boolean;
  size?: number | null;
}

export interface UiNavRoute {
  id: string;
  title?: string;
  child: UiNode;
}

export interface UiListItem {
  id: string;
  height?: number | null;
  child: UiNode;
}
export type UiSizeSpec = 'auto' | number | { px: number } | { fill: number };
export type UiInsets = number | { top?: number; right?: number; bottom?: number; left?: number };
export type UiAxis = 'row' | 'column';
export type UiAlign = 'start' | 'center' | 'end' | 'stretch';
export type UiStatVariant = 'compact' | 'sparkline' | 'range_bar';
export type UiPlotSeriesKind = 'line' | 'area' | 'stacked_area' | 'band' | 'candles' | 'histogram';
export type UiReadoutMode = 'header_bar' | 'floating';
export type UiTableAlign = 'left' | 'center' | 'right';
export type UiTableSortDir = 'asc' | 'desc';

/** Visual surface style (borders/shadows are opt-in; omitted fields fall back). */
export interface UiBoxStyle {
  bg?: number | null;
  bg_bottom?: number | null;
  hover_bg?: number | null;
  border?: number | null;
  border_width?: number;
  radius?: number | [number, number, number, number];
  shadow_color?: number | null;
  shadow_blur?: number;
  shadow_offset_y?: number;
  opacity?: number;
}

export interface UiNodeBaseOptions {
  /** Explicit node id; omit to receive a deterministic auto id (`_u1`, …). */
  id?: string;
  visible?: boolean;
  priority?: number;
  width?: UiSizeSpec;
  height?: UiSizeSpec;
  /** Shorthand for `width` / `height`. */
  w?: UiSizeSpec;
  h?: UiSizeSpec;
  fill?: boolean | number;
  /** Shorthand for `width`/`height` `{ fill }` (`flex` and `grow` are aliases). */
  flex?: number;
  grow?: number;
  minWidth?: number;
  minHeight?: number;
  padding?: UiInsets;
  /** Opt-in constraints / `Stack` positioning hints. */
  layout?: UiLayoutHints;
  /** Maximum size (aliases for `layout.max_*`). */
  maxWidth?: number;
  maxHeight?: number;
  aspectRatio?: number;
  flexShrink?: number;
  alignSelf?: UiAlign;
  /** Stack paint/hit order hint (lower paints first). */
  z?: number;
  /** Edge pinning inside a `Stack`/`Overlay`. */
  inset?: UiInsetHint;
}

/* ------------------------------------------------------------------
 * Exact serialized IR (`kind.type` discriminates `NodeKind`)
 * ------------------------------------------------------------------ */

export interface UiTabItem {
  label: string;
  child: UiNode;
}

export interface UiContainerKind {
  type: 'container';
  axis: UiAxis;
  align: UiAlign;
  gap: number;
  fold_width: number | null;
  scrollable: boolean;
  style: UiBoxStyle | null;
  children: UiNode[];
}

export interface UiGridKind {
  type: 'grid';
  columns: number;
  min_col_width: number;
  gap: number;
  style: UiBoxStyle | null;
  children: UiNode[];
}

export interface UiTabsKind {
  type: 'tabs';
  active: number;
  items: UiTabItem[];
}

export interface UiSurfaceKind {
  type: 'surface';
  style: UiBoxStyle;
  interactive: boolean;
  child: UiNode | null;
}

export interface UiTextKind {
  type: 'text';
  content: string;
  size: number;
  color: number | null;
  mono: boolean;
  wrap: boolean;
}

export interface UiRuleKind {
  type: 'rule';
  thickness: number;
  color: number | null;
}

export interface UiSpacerKind {
  type: 'spacer';
}

export interface UiThemeToggleKind {
  type: 'theme_toggle';
  size: number;
}

export interface UiSegmentedKind {
  type: 'segmented';
  active: number;
  options: string[];
}

export interface UiButtonKind {
  type: 'button';
  label: string;
  tone: UiTone;
}

export interface UiSwitchKind {
  type: 'switch';
  checked: boolean;
  label: string | null;
  tone: UiTone;
}

export interface UiInputKind {
  type: 'input';
  value: string;
  placeholder: string;
  tone: UiTone;
  keyboard: UiKeyboardHint;
  prefix_icon: string | null;
  clearable: boolean;
  error: string | null;
}

export interface UiStatKind {
  type: 'stat';
  label: string;
  badge: string | null;
  variant: UiStatVariant;
  value: number;
  value_text: string | null;
  prefix: string;
  suffix: string;
  precision: number | null;
  delta: string | null;
  tone: UiTone;
  caption: string | null;
  sparkline: number[];
  range_bounds: [number, number] | null;
  animate_value: boolean;
  interactive: boolean;
  border_width: number;
  border_color: number | null;
  shadow_blur: number;
  shadow_offset_y: number;
  update_seq: number;
}

export interface UiPlotKind {
  type: 'plot';
  title: string | null;
  subtitle: string | null;
  series: UiPlotSeries[];
  panes: UiPlotPane[];
  thresholds: UiPlotThreshold[];
  show_grid: boolean;
  show_legend: boolean;
  crosshair: boolean;
  readout_mode: UiReadoutMode;
  stacked_percent: boolean;
  y_prefix: string;
  y_suffix: string;
  y_precision: number | null;
  view_range: [number, number] | null;
  empty_text: string | null;
  border_width: number;
  border_color: number | null;
  shadow_blur: number;
  shadow_offset_y: number;
}

export interface UiTableKind {
  type: 'table';
  title: string | null;
  columns: UiTableColumn[];
  rows: UiTableRow[];
  sort_col: number | null;
  sort_dir: UiTableSortDir;
  selected_row: string | null;
  row_height: number;
  striped: boolean;
  empty_text: string | null;
  border_width: number;
  border_color: number | null;
  shadow_blur: number;
  shadow_offset_y: number;
}

export interface UiFeedKind {
  type: 'feed';
  title: string | null;
  items: UiFeedItem[];
  max_items: number;
  newest_first: boolean;
  empty_text: string | null;
  border_width: number;
  border_color: number | null;
  shadow_blur: number;
  shadow_offset_y: number;
  update_seq: number;
}

export interface UiGraphKind {
  type: 'graph';
  title: string | null;
  nodes: UiGraphNode[];
  edges: UiGraphEdge[];
  selected_node: string | null;
  empty_text: string | null;
  border_width: number;
  border_color: number | null;
  shadow_blur: number;
  shadow_offset_y: number;
}

export interface UiCodeKind {
  type: 'code_text';
  title: string | null;
  language: string | null;
  content: string;
  show_line_numbers: boolean;
  highlights: UiCodeHighlight[];
  border_width: number;
  border_color: number | null;
  shadow_blur: number;
  shadow_offset_y: number;
}

export interface UiRichTextKind {
  type: 'rich_text';
  spans: UiTextSpan[];
  size: number;
  wrap: boolean;
  line_height: number | null;
}

export interface UiIconKind {
  type: 'icon';
  name: string;
  size: number;
  color: number | null;
}

export interface UiImageKind {
  type: 'image';
  src: string;
  fit: UiImageFit;
  radius: [number, number, number, number];
  tint: number | null;
  placeholder: number | null;
  alt: string;
}

export interface UiStackKind {
  type: 'stack';
  align: UiStackAlign;
  gap: number;
  style: UiBoxStyle | null;
  children: UiNode[];
}

export interface UiSafeAreaKind {
  type: 'safe_area';
  insets: { top: number; right: number; bottom: number; left: number };
  keyboard_avoidance: boolean;
  keyboard_height: number;
  child: UiNode | null;
}

export interface UiOverlayKind {
  type: 'overlay';
  scrim: number | null;
  dismissable: boolean;
  align: UiStackAlign;
  style: UiBoxStyle | null;
  child: UiNode | null;
}

export interface UiNavStackKind {
  type: 'nav_stack';
  routes: UiNavRoute[];
  active: number;
  progress: number;
  swipe_back: boolean;
}

export interface UiListViewKind {
  type: 'list_view';
  items: UiListItem[];
  estimated_height: number;
  spacing: number;
  overscan: number;
  style: UiBoxStyle | null;
}

export interface UiTextAreaKind {
  type: 'text_area';
  value: string;
  placeholder: string;
  rows: number;
  tone: UiTone;
  secure: boolean;
  max_len: number;
  keyboard: UiKeyboardHint;
}

export interface UiSliderKind {
  type: 'slider';
  value: number;
  min: number;
  max: number;
  step: number;
  tone: UiTone;
  label: string | null;
  show_value: boolean;
}

export interface UiRangeSliderKind {
  type: 'range_slider';
  low: number;
  high: number;
  min: number;
  max: number;
  step: number;
  tone: UiTone;
  label: string | null;
  show_value: boolean;
}

export type UiProgressVariant = 'linear' | 'circular';

export interface UiProgressKind {
  type: 'progress';
  value: number;
  variant: UiProgressVariant;
  tone: UiTone;
  label: string | null;
  indeterminate: boolean;
}

export interface UiPressableKind {
  type: 'pressable';
  hit_slop: number;
  disabled: boolean;
  child: UiNode | null;
}

export interface UiDraggableKind {
  type: 'draggable';
  axis: UiDragAxis;
  offset: [number, number];
  inertia: boolean;
  child: UiNode | null;
}

/** Exact serialized `NodeKind` union (matches `crates/sofuu-ui/src/ir.rs`). */
export type UiNodeKind =
  | UiContainerKind
  | UiGridKind
  | UiTabsKind
  | UiSurfaceKind
  | UiTextKind
  | UiRuleKind
  | UiSpacerKind
  | UiThemeToggleKind
  | UiSegmentedKind
  | UiButtonKind
  | UiSwitchKind
  | UiInputKind
  | UiStatKind
  | UiPlotKind
  | UiTableKind
  | UiFeedKind
  | UiGraphKind
  | UiCodeKind
  | UiRichTextKind
  | UiIconKind
  | UiImageKind
  | UiStackKind
  | UiSafeAreaKind
  | UiOverlayKind
  | UiNavStackKind
  | UiListViewKind
  | UiTextAreaKind
  | UiSliderKind
  | UiRangeSliderKind
  | UiProgressKind
  | UiPressableKind
  | UiDraggableKind;

export interface UiNode {
  id: string;
  priority: number;
  visible: boolean;
  width: 'auto' | { px: number } | { fill: number };
  height: 'auto' | { px: number } | { fill: number };
  min_width: number;
  min_height: number;
  padding: { top: number; right: number; bottom: number; left: number };
  /** Opt-in `sofuu.fx` specs attached by the decorator builders. */
  fx?: UiFxSpec[];
  /** Extra constraints / `Stack` positioning hints. */
  layout?: UiLayoutHints;
  kind: UiNodeKind;
  /** Present on builder-returned handles; returns the plain serializable node. */
  toNode?(): UiNode;
  setVisible?(visible: boolean): UiNode;
}

export interface UiScene {
  title: string;
  theme: UiThemeMode;
  root: UiNode;
}

/* ------------------------------------------------------------------
 * Component data shapes (exact wire form)
 * ------------------------------------------------------------------ */

export interface UiPlotPoint {
  x: number;
  y: number;
  y2?: number | null;
  label?: string | null;
}

export interface UiCandlePoint {
  x: number;
  open: number;
  high: number;
  low: number;
  close: number;
  label?: string | null;
}

export interface UiPlotThreshold {
  value: number;
  label?: string | null;
  color?: number | null;
}

/** Exact serialized series (the builder fills `name`/`kind` defaults). */
export interface UiPlotSeries {
  id: string;
  name: string;
  kind: UiPlotSeriesKind;
  color: number | null;
  secondary_color: number | null;
  visible: boolean;
  dashed: boolean;
  stroke_width: number | null;
  points: UiPlotPoint[];
  candles: UiCandlePoint[];
}

/** Builder-friendly series input (`secondaryColor`/`strokeWidth` accepted). */
export interface UiPlotSeriesInput {
  id: string;
  name?: string;
  kind?: UiPlotSeriesKind;
  color?: number | null;
  secondaryColor?: number | null;
  secondary_color?: number | null;
  visible?: boolean;
  dashed?: boolean;
  strokeWidth?: number | null;
  stroke_width?: number | null;
  points?: UiPlotPoint[];
  candles?: UiCandlePoint[];
}

export interface UiPlotPane {
  id: string;
  label: string | null;
  height_weight: number;
  series: UiPlotSeries[];
  thresholds: UiPlotThreshold[];
  stacked_percent: boolean;
  y_prefix: string;
  y_suffix: string;
  y_precision: number | null;
}

export interface UiPlotPaneInput {
  id?: string;
  label?: string | null;
  heightWeight?: number;
  weight?: number;
  series?: UiPlotSeriesInput[];
  thresholds?: UiPlotThreshold[];
  percent?: boolean;
  yPrefix?: string;
  ySuffix?: string;
  yPrecision?: number | null;
}

export interface UiTableColumn {
  key: string;
  label: string;
  width: number | null;
  align: UiTableAlign;
  sortable: boolean;
}

export interface UiTableColumnInput {
  key: string;
  label?: string;
  width?: number | null;
  align?: UiTableAlign;
  sortable?: boolean;
}

export interface UiTableCell {
  text: string;
  tone: UiTone | null;
  badge: boolean;
}

/** Builder-friendly cell input (tone/badge optional). */
export interface UiTableCellInput {
  text: string;
  tone?: UiTone | null;
  badge?: boolean;
}

export interface UiTableRow {
  id: string;
  tone: UiTone | null;
  cells: UiTableCell[];
}

/** Builder input: cells may be plain strings/numbers and are normalized. */
export interface UiTableRowInput {
  id: string;
  tone?: UiTone | null;
  cells: Array<string | number | UiTableCellInput>;
}

export interface UiFeedItem {
  id: string;
  time?: string;
  badge?: string | null;
  tone?: UiTone;
  title: string;
  body?: string | null;
}

export interface UiGraphNode {
  id: string;
  label: string;
  subtitle?: string | null;
  tone?: UiTone;
  pos?: [number, number] | null;
}

export interface UiGraphEdge {
  from: string;
  to: string;
  label?: string | null;
  active?: boolean;
  tone?: UiTone | null;
}

export interface UiCodeHighlight {
  line: number;
  tone: UiTone;
}

export type UiEvent =
  | { type: 'click'; id: string }
  | { type: 'tab_changed'; id: string; index: number }
  | { type: 'segmented_changed'; id: string; index: number }
  | { type: 'theme_changed'; id: string; mode: UiThemeMode }
  | { type: 'switch_toggled'; id: string; checked: boolean }
  | { type: 'text_changed'; id: string; value: string }
  | { type: 'input_submitted'; id: string; value: string }
  | { type: 'table_sorted'; id: string; col_index: number; asc: boolean }
  | { type: 'row_selected'; id: string; row_id: string }
  | { type: 'feed_item_clicked'; id: string; item_id: string }
  | { type: 'graph_node_clicked'; id: string; node_id: string }
  | { type: 'series_toggled'; id: string; series_index: number }
  | { type: 'scrolled'; id: string; offset_y: number }
  | { type: 'value_changed'; id: string; value: number }
  | { type: 'range_changed'; id: string; low: number; high: number }
  | { type: 'list_item_clicked'; id: string; index: number; item_id: string }
  | { type: 'back_requested'; id: string; route: string }
  | { type: 'refresh_requested'; id: string }
  | { type: 'dragged'; id: string; dx: number; dy: number }
  | { type: 'copy_requested'; id: string; text: string }
  | { type: 'window_closed' };

/* ------------------------------------------------------------------
 * Exact serialized `ScenePatch` union (`op` discriminates)
 * ------------------------------------------------------------------ */

export type UiScenePatch =
  | ({ op: 'replace_scene' } & UiScene)
  | { op: 'set_theme'; mode: UiThemeMode }
  | { op: 'set_text'; id: string; content: string }
  | { op: 'set_visible'; id: string; visible: boolean }
  | { op: 'set_tab'; id: string; active: number }
  | { op: 'set_segmented'; id: string; active: number }
  | { op: 'set_switch'; id: string; checked: boolean }
  | { op: 'set_input'; id: string; value: string }
  | { op: 'sort_table'; id: string; col_index: number }
  | { op: 'select_table_row'; id: string; row_id: string | null }
  | { op: 'append_feed_item'; id: string; item: UiFeedItem; max_items?: number }
  | { op: 'select_graph_node'; id: string; node_id: string | null }
  | { op: 'set_graph_node_tone'; id: string; node_id: string; tone: UiTone }
  | { op: 'set_graph_edge_active'; id: string; from: string; to: string; active: boolean }
  | { op: 'set_code'; id: string; content: string; highlights?: UiCodeHighlight[] }
  | { op: 'replace_node'; id: string; node: UiNode }
  | { op: 'set_table_rows'; id: string; rows: UiTableRow[]; selected_row?: string | null }
  | {
      op: 'set_plot_data';
      id: string;
      title?: string | null;
      subtitle?: string | null;
      series?: UiPlotSeries[];
      panes?: UiPlotPane[];
      thresholds?: UiPlotThreshold[];
    }
  | {
      op: 'set_graph';
      id: string;
      title?: string | null;
      nodes: UiGraphNode[];
      edges: UiGraphEdge[];
      selected_node?: string | null;
    }
  | { op: 'set_children'; id: string; children: UiNode[] }
  | { op: 'set_list_items'; id: string; items: UiListItem[] }
  | { op: 'set_rich_text'; id: string; spans: UiTextSpan[] }
  | { op: 'set_icon'; id: string; name: string; color?: number | null }
  | { op: 'set_image'; id: string; src: string }
  | { op: 'set_slider'; id: string; value: number }
  | { op: 'set_range_slider'; id: string; low: number; high: number }
  | { op: 'set_progress'; id: string; value: number; indeterminate?: boolean }
  | { op: 'set_text_area'; id: string; value: string }
  | { op: 'set_draggable'; id: string; offset: [number, number] }
  | { op: 'set_nav_progress'; id: string; progress: number }
  | { op: 'nav_push'; id: string; route: UiNavRoute }
  | { op: 'nav_pop'; id: string }
  | { op: 'nav_replace'; id: string; route: UiNavRoute }
  | { op: 'nav_pop_to_root'; id: string }
  | {
      op: 'update_stat';
      id: string;
      value?: number;
      value_text?: string | null;
      delta?: string | null;
      tone?: UiTone;
      caption?: string | null;
      push_sparkline?: number;
      sparkline_max_len?: number;
    }
  | { op: 'append_plot_point'; id: string; series_id: string; point: UiPlotPoint; max_points?: number }
  | { op: 'upsert_candle'; id: string; series_id: string; candle: UiCandlePoint; max_candles?: number }
  | { op: 'toggle_plot_series'; id: string; series_index: number }
  | { op: 'zoom_plot'; id: string; delta_y: number; anchor_frac: number }
  | { op: 'pan_plot'; id: string; delta_frac: number }
  | { op: 'reset_plot_view'; id: string };

/* ------------------------------------------------------------------
 * Builder options
 * ------------------------------------------------------------------ */

export interface UiContainerOptions extends UiNodeBaseOptions {
  align?: UiAlign;
  gap?: number;
  foldBelow?: number | null;
  scrollable?: boolean;
  style?: UiBoxStyle | null;
  columns?: number;
  minColWidth?: number;
}

export interface UiTabsOptions extends UiNodeBaseOptions {
  active?: number;
  onChange?: (index: number, label: string, event: UiEvent) => void;
}

export interface UiTextOptions extends UiNodeBaseOptions {
  size?: number;
  color?: number | null;
  mono?: boolean;
  wrap?: boolean;
}

export interface UiRuleOptions extends UiNodeBaseOptions {
  thickness?: number;
  color?: number | null;
}

export interface UiThemeToggleOptions extends UiNodeBaseOptions {
  size?: number;
}

export interface UiSegmentedOptions extends UiNodeBaseOptions {
  onChange?: (index: number, label: string, event: UiEvent) => void;
}

export interface UiButtonOptions extends UiNodeBaseOptions {
  tone?: UiTone;
  onClick?: (event: UiEvent) => void;
}

export interface UiSwitchOptions extends UiNodeBaseOptions {
  tone?: UiTone;
  onChange?: (checked: boolean, event: UiEvent) => void;
}

export interface UiInputOptions extends UiNodeBaseOptions {
  value?: string;
  tone?: UiTone;
  keyboard?: UiKeyboardHint;
  inputMode?: UiKeyboardHint;
  inputType?: UiKeyboardHint;
  /** Leading SDF icon name (`search`, `mail`, …). */
  prefixIcon?: string;
  /** Show a `×` affordance that clears the field. */
  clearable?: boolean;
  /** Inline validation caption under the field (danger tone). */
  error?: string | null;
  onChange?: (value: string, event: UiEvent) => void;
  onSubmit?: (value: string, event: UiEvent) => void;
}

export interface UiRichTextOptions extends UiNodeBaseOptions {
  size?: number;
  wrap?: boolean;
  lineHeight?: number;
  onChange?: (event: UiEvent) => void;
}

export interface UiIconOptions extends UiNodeBaseOptions {
  size?: number;
  color?: number | null;
}

export interface UiImageOptions extends UiNodeBaseOptions {
  fit?: UiImageFit;
  radius?: number | [number, number] | [number, number, number, number];
  tint?: number | null;
  placeholder?: number | null;
  alt?: string;
}

export interface UiStackOptions extends UiNodeBaseOptions {
  align?: UiStackAlign;
  gap?: number;
  style?: UiBoxStyle | null;
  z?: number;
}

export interface UiSafeAreaOptions extends UiNodeBaseOptions {
  insets?: UiInsetHint;
  keyboardAvoidance?: boolean;
  keyboardHeight?: number;
}

export interface UiOverlayOptions extends UiNodeBaseOptions {
  scrim?: number | null;
  dismissable?: boolean;
  align?: UiStackAlign;
  style?: UiBoxStyle | null;
  onClick?: (event: UiEvent) => void;
}

export interface UiNavStackOptions extends UiNodeBaseOptions {
  active?: number;
  swipeBack?: boolean;
  onBack?: (route: string, event: UiEvent) => void;
}

export interface UiListViewOptions extends UiNodeBaseOptions {
  estimatedHeight?: number;
  spacing?: number;
  overscan?: number;
  style?: UiBoxStyle | null;
  onItemClick?: (index: number, itemId: string, event: UiEvent) => void;
  onRefresh?: (event: UiEvent) => void;
}

export interface UiTextAreaOptions extends UiNodeBaseOptions {
  value?: string;
  rows?: number;
  tone?: UiTone;
  secure?: boolean;
  maxLen?: number;
  keyboard?: UiKeyboardHint;
  inputMode?: UiKeyboardHint;
  inputType?: UiKeyboardHint;
  /** Leading SDF icon name (`search`, `mail`, …). */
  prefixIcon?: string;
  /** Show a `×` affordance that clears the field. */
  clearable?: boolean;
  /** Inline validation caption under the field (danger tone). */
  error?: string | null;
  onChange?: (value: string, event: UiEvent) => void;
  onSubmit?: (value: string, event: UiEvent) => void;
}

export interface UiSliderOptions extends UiNodeBaseOptions {
  value?: number;
  min?: number;
  max?: number;
  step?: number;
  tone?: UiTone;
  label?: string | null;
  showValue?: boolean;
  onChange?: (value: number, event: UiEvent) => void;
}

export interface UiRangeSliderOptions extends UiNodeBaseOptions {
  low?: number;
  high?: number;
  min?: number;
  max?: number;
  step?: number;
  tone?: UiTone;
  label?: string | null;
  showValue?: boolean;
  onChange?: (low: number, high: number, event: UiEvent) => void;
}

export interface UiProgressOptions extends UiNodeBaseOptions {
  /** Fraction `0..1`, or `0..100` when greater than 1. */
  value?: number;
  variant?: UiProgressVariant;
  tone?: UiTone;
  label?: string | null;
  indeterminate?: boolean;
}

export interface UiPressableOptions extends UiNodeBaseOptions {
  hitSlop?: number;
  disabled?: boolean;
  onClick?: (event: UiEvent) => void;
}

export interface UiDraggableOptions extends UiNodeBaseOptions {
  axis?: UiDragAxis;
  offset?: [number, number];
  inertia?: boolean;
  onDrag?: (dx: number, dy: number, event: UiEvent) => void;
}

export interface UiStatOptions extends UiNodeBaseOptions {
  variant?: UiStatVariant;
  valueText?: string | null;
  prefix?: string;
  suffix?: string;
  precision?: number | null;
  badge?: string | null;
  delta?: string | null;
  caption?: string | null;
  tone?: UiTone;
  sparkline?: number[];
  range?: [number, number];
  interactive?: boolean;
  onClick?: (event: UiEvent) => void;
}

export interface UiStatUpdate {
  value?: number;
  valueText?: string | null;
  delta?: string | null;
  tone?: UiTone;
  caption?: string | null;
  pushSparkline?: number;
  sparklineMaxLen?: number;
}

export interface UiPlotOptions extends UiNodeBaseOptions {
  title?: string | null;
  subtitle?: string | null;
  panes?: UiPlotPane[];
  series?: UiPlotSeriesInput[];
  thresholds?: UiPlotThreshold[];
  showGrid?: boolean;
  showCrosshair?: boolean;
  crosshair?: boolean;
  showLegend?: boolean;
  readoutMode?: UiReadoutMode;
  percent?: boolean;
  yPrefix?: string;
  ySuffix?: string;
  yPrecision?: number | null;
  emptyText?: string | null;
}

export interface UiPlotDataInput {
  title?: string | null;
  subtitle?: string | null;
  series?: UiPlotSeriesInput[];
  panes?: UiPlotPane[];
  thresholds?: UiPlotThreshold[];
}

export interface UiTableOptions extends UiNodeBaseOptions {
  title?: string | null;
  columns?: UiTableColumnInput[];
  rows?: UiTableRowInput[];
  selectedRow?: string | null;
  rowHeight?: number;
  striped?: boolean;
  emptyText?: string | null;
  onSelect?: (rowId: string, event: UiEvent) => void;
  onSort?: (colIndex: number, asc: boolean, event: UiEvent) => void;
}

export interface UiFeedOptions extends UiNodeBaseOptions {
  title?: string | null;
  items?: UiFeedItem[];
  maxItems?: number;
  newestFirst?: boolean;
  emptyText?: string | null;
  onItemClick?: (itemId: string, event: UiEvent) => void;
}

export interface UiGraphOptions extends UiNodeBaseOptions {
  title?: string | null;
  nodes?: UiGraphNode[];
  edges?: UiGraphEdge[];
  selectedNode?: string | null;
  emptyText?: string | null;
  onNodeClick?: (nodeId: string, event: UiEvent) => void;
}

export interface UiGraphDataInput {
  title?: string | null;
  nodes?: UiGraphNode[];
  edges?: UiGraphEdge[];
  selectedNode?: string | null;
}

export interface UiCodeOptions extends UiNodeBaseOptions {
  title?: string | null;
  language?: string | null;
  lineNumbers?: boolean;
  highlights?: UiCodeHighlight[];
}

/* ------------------------------------------------------------------
 * Component handles (the returned node, augmented with methods)
 * ------------------------------------------------------------------ */

export interface UiNodeHandle extends UiNode {
  toNode(): UiNode;
  setVisible(visible: boolean): this;
}

export interface UiTextHandle extends UiNodeHandle {
  kind: UiTextKind;
  toNode(): UiTextHandle;
  set(content: string): this;
}

export interface UiContainerHandle extends UiNodeHandle {
  kind: UiContainerKind | UiGridKind;
  toNode(): UiContainerHandle;
  setChildren(children: UiNode[]): this;
}

export interface UiTabsHandle extends UiNodeHandle {
  kind: UiTabsKind;
  toNode(): UiTabsHandle;
  setActive(index: number): this;
}

export interface UiSegmentedHandle extends UiNodeHandle {
  kind: UiSegmentedKind;
  toNode(): UiSegmentedHandle;
  setActive(index: number): this;
}

export interface UiButtonHandle extends UiNodeHandle {
  kind: UiButtonKind;
  toNode(): UiButtonHandle;
}

export interface UiSwitchHandle extends UiNodeHandle {
  kind: UiSwitchKind;
  readonly checked: boolean;
  toNode(): UiSwitchHandle;
  setChecked(checked: boolean): this;
}

export interface UiInputHandle extends UiNodeHandle {
  kind: UiInputKind;
  toNode(): UiInputHandle;
  setValue(value: string): this;
}

export interface UiStatHandle extends UiNodeHandle {
  kind: UiStatKind;
  toNode(): UiStatHandle;
  update(fields: UiStatUpdate): this;
}

export interface UiPlotHandle extends UiNodeHandle {
  kind: UiPlotKind;
  toNode(): UiPlotHandle;
  append(seriesId: string, point: UiPlotPoint, maxPoints?: number): this;
  upsertCandle(seriesId: string, candle: UiCandlePoint, maxCandles?: number): this;
  setData(data: UiPlotDataInput): this;
}

export interface UiTableHandle extends UiNodeHandle {
  kind: UiTableKind;
  toNode(): UiTableHandle;
  setRows(rows: UiTableRowInput[], selectedRow?: string | null): this;
  selectRow(rowId: string | null): this;
}

export interface UiFeedHandle extends UiNodeHandle {
  kind: UiFeedKind;
  toNode(): UiFeedHandle;
  append(item: UiFeedItem, maxItems?: number): this;
}

export interface UiGraphHandle extends UiNodeHandle {
  kind: UiGraphKind;
  toNode(): UiGraphHandle;
  setGraph(data: UiGraphDataInput): this;
  selectNode(nodeId: string | null): this;
}

export interface UiCodeHandle extends UiNodeHandle {
  kind: UiCodeKind;
  toNode(): UiCodeHandle;
  setContent(content: string, highlights?: UiCodeHighlight[]): this;
}

export interface UiRichTextHandle extends UiNodeHandle {
  kind: UiRichTextKind;
  toNode(): UiRichTextHandle;
  setSpans(spans: Array<string | UiTextSpan>): this;
}

export interface UiIconHandle extends UiNodeHandle {
  kind: UiIconKind;
  toNode(): UiIconHandle;
  setIcon(name: string, color?: number | null): this;
}

export interface UiImageHandle extends UiNodeHandle {
  kind: UiImageKind;
  toNode(): UiImageHandle;
  setSrc(src: string): this;
}

export interface UiStackHandle extends UiNodeHandle {
  kind: UiStackKind;
  toNode(): UiStackHandle;
  setChildren(children: UiNode[]): this;
}

export interface UiNavStackHandle extends UiNodeHandle {
  kind: UiNavStackKind;
  toNode(): UiNavStackHandle;
  push(route: UiNavRoute): this;
  pop(): this;
  replace(route: UiNavRoute): this;
  popToRoot(): this;
  setProgress(progress: number): this;
}

export interface UiListViewHandle extends UiNodeHandle {
  kind: UiListViewKind;
  toNode(): UiListViewHandle;
  setItems(items: UiListItem[]): this;
}

export interface UiTextAreaHandle extends UiNodeHandle {
  kind: UiTextAreaKind;
  toNode(): UiTextAreaHandle;
  setValue(value: string): this;
}

export interface UiSliderHandle extends UiNodeHandle {
  kind: UiSliderKind;
  toNode(): UiSliderHandle;
  setValue(value: number): this;
}

export interface UiRangeSliderHandle extends UiNodeHandle {
  kind: UiRangeSliderKind;
  toNode(): UiRangeSliderHandle;
  setValue(low: number, high?: number): this;
}

export interface UiProgressHandle extends UiNodeHandle {
  kind: UiProgressKind;
  toNode(): UiProgressHandle;
  setValue(value: number, indeterminate?: boolean): this;
}

export interface UiDraggableHandle extends UiNodeHandle {
  kind: UiDraggableKind;
  toNode(): UiDraggableHandle;
  setOffset(x: number, y: number): this;
}

/* ------------------------------------------------------------------
 * View / scene
 * ------------------------------------------------------------------ */

export interface UiTriggerLike {
  start(): Promise<unknown>;
  stop(reason?: string): void;
  gate?(payload: unknown, ctx?: unknown): boolean;
}

export type UiBindTriggerResult = Promise<Array<PromiseSettledResult<unknown>>> & {
  isPaused(): boolean;
};

export interface UiView {
  readonly id: string;
  mount(sceneOrNode: UiScene | UiNode | { toNode(): UiNode }): UiView;
  patch(patch: UiScenePatch): number;
  setTheme(mode: UiThemeMode): number;
  setText(nodeId: string, content: string): number;
  setVisible(nodeId: string, visible: boolean): number;
  updateStat(
    nodeId: string,
    fields: {
      value?: number;
      valueText?: string | null;
      delta?: string | null;
      tone?: UiTone;
      caption?: string | null;
      pushSparkline?: number;
      sparklineMaxLen?: number;
    },
  ): number;
  appendPlot(plotId: string, seriesId: string, point: UiPlotPoint, maxPoints?: number): number;
  upsertCandle(plotId: string, seriesId: string, candle: UiCandlePoint, maxCandles?: number): number;
  appendFeed(feedId: string, item: UiFeedItem, maxItems?: number): number;
  setCode(codeId: string, content: string, highlights?: UiCodeHighlight[]): number;
  setSwitch(switchId: string, checked: boolean): number;
  setInput(inputId: string, value: string): number;
  /** Replace a node's complete effect list (low-level bridge for `sofuu.fx`). */
  setFx(nodeId: string, specs: UiFxSpec[]): number;
  /** Fire a one-shot impulse (`confetti`, `spark_burst`, `shake`). */
  triggerFx(
    kind: UiFxKind,
    opts?: { target?: UiNode | string; x?: number; y?: number; count?: number; seed?: number },
  ): number;
  /** Reactive `voice_orb` amplitude without rebuilding the scene. */
  setFxAmplitude(nodeId: string, value: number): number;
  setSegmented(nodeId: string, active: number): number;
  setTab(nodeId: string, active: number): number;
  selectTableRow(tableId: string, rowId: string | null): number;
  setTableRows(tableId: string, rows: UiTableRowInput[], selectedRow?: string | null): number;
  setPlotData(plotId: string, data: UiPlotDataInput): number;
  setGraph(graphId: string, data: UiGraphDataInput): number;
  setChildren(nodeId: string, children: UiNode[]): number;
  /**
   * Wire a reactive trigger loop to the window lifecycle: stops the runner on
   * `window_closed`, optionally pauses/resumes the gate from a live switch,
   * and resolves once both `runner.start()` and the window settle.
   */
  bindTrigger(
    runner: UiTriggerLike,
    binds?: { liveSwitch?: UiSwitchHandle },
  ): UiBindTriggerResult;
  onEvent(listener: (event: UiEvent) => void): UiView;
  waitUntilClosed(): Promise<void>;
}

export interface UiAppBuilder {
  dark(): UiAppBuilder;
  light(): UiAppBuilder;
  header(node: UiNode): UiAppBuilder;
  stats(list: UiNode[]): UiAppBuilder;
  add(node: UiNode): UiAppBuilder;
  build(): UiScene;
  mount(): UiView;
}

/** Dependency-free observable cell returned by `ui.signal`. */
export interface UiSignal<T> {
  get(): T;
  set(next: T): T;
  update(fn: (prev: T) => T): T;
  /** Returns an unsubscribe function. `immediate` fires once with the current value. */
  subscribe(fn: (value: T) => void, immediate?: boolean): () => void;
}

/* ------------------------------------------------------------------
 * GPU UI effects & choreography (`sofuu.fx`)
 * ------------------------------------------------------------------ */

export type UiFxSpring = 'snappy' | 'default' | 'gentle';

export type UiFxKind =
  | 'enter'
  | 'fade'
  | 'slide'
  | 'scale_pop'
  | 'shake'
  | 'number_roll'
  | 'shared'
  | 'shimmer'
  | 'touch_ripple'
  | 'border_beam'
  | 'cursor_glow'
  | 'aurora'
  | 'pulse_ring'
  | 'voice_orb'
  | 'backdrop_blur'
  | 'text_shimmer'
  | 'text_scramble'
  | 'typewriter'
  | 'confetti'
  | 'spark_burst';

/** Exact wire shape of one opt-in effect (`Node.fx`). */
export interface UiFxSpec {
  kind: UiFxKind;
  preset?: string;
  spring?: UiFxSpring;
  delay_ms?: number;
  duration_ms?: number;
  slide_x?: number;
  slide_y?: number;
  scale?: number;
  from?: number;
  to?: number;
  precision?: number;
  prefix?: string;
  suffix?: string;
  tone?: UiTone;
  color0?: number;
  color1?: number;
  color2?: number;
  speed?: number;
  intensity?: number;
  radius?: number;
  width?: number;
  amplitude?: number;
  active?: boolean;
  count?: number;
  auto?: boolean;
  shared_key?: string | null;
  blur?: number;
  rim?: number;
  seed?: number;
  exit?: boolean;
}

/** Camel-case authoring options accepted by the `sofuu.fx` decorators. */
export interface UiFxOptions {
  preset?: string;
  spring?: UiFxSpring;
  delayMs?: number;
  durationMs?: number;
  slideX?: number;
  slideY?: number;
  scale?: number;
  from?: number;
  to?: number;
  precision?: number;
  prefix?: string;
  suffix?: string;
  tone?: UiTone;
  color0?: number;
  color1?: number;
  color2?: number;
  speed?: number;
  intensity?: number;
  radius?: number;
  width?: number;
  amplitude?: number;
  active?: boolean;
  count?: number;
  auto?: boolean;
  sharedKey?: string;
  blur?: number;
  rim?: number;
  seed?: number;
  exit?: boolean;
  /** `voiceOrb` convenience: rendered node edge in logical px. */
  size?: number;
  baseDelayMs?: number;
}

export interface SofuuFx {
  spec(kind: UiFxKind, opts?: UiFxOptions): UiFxSpec;

  enter<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  fade<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  slide<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  scalePop<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  stagger<T extends UiNode>(items: T[], opts?: UiFxOptions): T[];
  shared<T extends UiNode>(key: string, node: T): T;
  shared<T extends UiNode>(node: T, key: string): T;
  shake<T extends UiNode>(node: T): T;

  numberRoll<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  typewriter<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  scramble<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  textShimmer<T extends UiNode>(node: T, opts?: UiFxOptions): T;

  shimmer<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  touchRipple<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  borderBeam<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  cursorGlow<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  aurora<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  pulseRing<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  backdropBlur<T extends UiNode>(node: T, opts?: UiFxOptions): T;
  voiceOrb(opts: UiFxOptions & { amplitude?: number }): UiNode;
  voiceOrb<T extends UiNode>(node: T, opts?: UiFxOptions): T;

  burst(
    kind: 'confetti' | 'spark_burst' | 'sparkBurst' | 'shake' | UiFxKind,
    opts?: {
      origin?: UiNode | { x: number; y: number };
      target?: UiNode | string;
      x?: number;
      y?: number;
      count?: number;
      seed?: number;
    },
  ): unknown;
  trigger: SofuuFx['burst'];
  confetti(origin?: UiNode | { x: number; y: number }, opts?: UiFxOptions): unknown;
  sparkBurst(origin?: UiNode | { x: number; y: number }, opts?: UiFxOptions): unknown;

  set<T extends UiNode>(node: T, specs: UiFxSpec[]): T;
  clear<T extends UiNode>(node: T): T;

  motion: {
    fade: SofuuFx['fade'];
    slide: SofuuFx['slide'];
    scalePop: SofuuFx['scalePop'];
    enter: SofuuFx['enter'];
    stagger: SofuuFx['stagger'];
    shake: SofuuFx['shake'];
    shared: SofuuFx['shared'];
  };
  text: {
    numberRoll: SofuuFx['numberRoll'];
    typewriter: SofuuFx['typewriter'];
    scramble: SofuuFx['scramble'];
    shimmer: SofuuFx['textShimmer'];
  };
  surface: {
    shimmer: SofuuFx['shimmer'];
    borderBeam: SofuuFx['borderBeam'];
    cursorGlow: SofuuFx['cursorGlow'];
    aurora: SofuuFx['aurora'];
    backdropBlur: SofuuFx['backdropBlur'];
  };
  bursts: {
    ripple: SofuuFx['touchRipple'];
    pulseRing: SofuuFx['pulseRing'];
    confetti: SofuuFx['confetti'];
    sparkBurst: SofuuFx['sparkBurst'];
    voiceOrb: SofuuFx['voiceOrb'];
  };
}

export interface SofuuUiKit {
  tokens: {
    tones: Record<string, number>;
    color(rgb: number, alpha?: number): number;
    tone(name: string, alpha?: number): number;
    radius(value?: number | number[]): [number, number, number, number];
    modes: string[];
  };
  Card(id: string, body: UiNode | UiNode[] | null, opts?: UiStackOptions & {
    title?: string;
    titleSize?: number;
    padding?: number;
    gap?: number;
    action?: UiNode;
  }): UiNodeHandle;
  ListTile(id: string, opts: UiPressableOptions & {
    title: string;
    subtitle?: string;
    icon?: string;
    leading?: UiNode;
    trailing?: UiNode;
    chevron?: boolean;
    titleSize?: number;
    subtitleSize?: number;
    padding?: number;
    iconColor?: number;
  }): UiNodeHandle;
  Badge(id: string, text: string, opts?: UiNodeBaseOptions & {
    tone?: UiTone; bg?: number; color?: number; size?: number; padding?: number;
  }): UiNodeHandle;
  Chip(id: string, text: string, opts?: UiPressableOptions & {
    selected?: boolean; tone?: UiTone; padding?: number;
  }): UiNodeHandle;
  Avatar(id: string, opts?: UiNodeBaseOptions & {
    src?: string; name?: string; initials?: string; size?: number; tone?: UiTone; alt?: string;
  }): UiNodeHandle;
  Dialog(id: string, opts?: UiOverlayOptions & {
    title?: string; body?: UiNode | UiNode[] | null;
    actions?: Array<{ label: string; tone?: UiTone; width?: UiSizeSpec; onClick?: (event: UiEvent) => void }>;
    visible?: boolean; width?: number; radius?: number; onDismiss?: (event: UiEvent) => void;
  }): UiNodeHandle;
  Toast(id: string, text: string, opts?: UiNodeBaseOptions & {
    tone?: UiTone; bg?: number; color?: number; duration?: number; visible?: boolean; width?: number;
  }): UiNodeHandle;
  Sheet(id: string, body: UiNode | UiNode[] | null, opts?: UiOverlayOptions & {
    title?: string; visible?: boolean; width?: number; onDismiss?: (event: UiEvent) => void;
  }): UiNodeHandle;
  Accordion(id: string, items: Array<{ title: string; body?: UiNode | UiNode[]; open?: boolean }>, opts?: {
    gap?: number;
  }): UiContainerHandle;
  BottomNav(id: string, items: Array<{ icon?: string; label?: string }>, opts?: {
    active?: number; activeTone?: UiTone; inactiveColor?: number | null; height?: number;
    bg?: number; border?: number; radius?: number[];
    onChange?: (index: number, item: { icon?: string; label?: string }, event: UiEvent) => void;
  }): UiContainerHandle;
  Sidebar(id: string, items: Array<{ label: string; icon?: string; active?: boolean; badge?: UiNode }>, opts?: {
    width?: number; bg?: number; border?: number; radius?: number[]; iconColor?: number;
    onSelect?: (index: number, item: { label: string; icon?: string; active?: boolean }, event: UiEvent) => void;
  }): UiContainerHandle;
  AppBar(id: string, opts?: {
    title?: string; titleSize?: number; leading?: UiNode; back?: boolean; actions?: UiNode[];
    height?: number; bg?: number; border?: number; radius?: number[]; shadowBlur?: number;
    onBack?: (event: UiEvent) => void;
  }): UiContainerHandle;
  Checkbox(id: string, label?: string, opts?: UiPressableOptions & {
    checked?: boolean; tone?: UiTone; radius?: number; onChange?: (checked: boolean, event: UiEvent) => void;
  }): UiNodeHandle & { checked(): boolean; setChecked(value: boolean): UiNodeHandle };
  Radio(id: string, label?: string, opts?: UiPressableOptions & {
    checked?: boolean; tone?: UiTone; onChange?: (checked: boolean, event: UiEvent) => void;
  }): UiNodeHandle & { checked(): boolean };
  Slider(id: string, opts?: UiSliderOptions): UiSliderHandle;
  RangeSlider(id: string, opts?: UiRangeSliderOptions): UiRangeSliderHandle;
  Progress(id: string, opts?: UiProgressOptions): UiProgressHandle;
  Spinner(id: string, opts?: UiNodeBaseOptions & { tone?: UiTone; size?: number }): UiProgressHandle;
  Select(id: string, options: Array<{ label: string; value?: string }>, opts?: {
    active?: number; placeholder?: string; width?: number; radius?: number; menuWidth?: number;
    bg?: number; border?: number; menuBg?: number;
    onChange?: (index: number, option: { label: string; value?: string }, event: UiEvent) => void;
  }): UiNodeHandle & { setValue(index: number): UiNodeHandle };
  Skeleton(id: string, opts?: UiNodeBaseOptions & {
    width?: UiSizeSpec; height?: number; radius?: number; bg?: number; intensity?: number;
  }): UiNodeHandle;
  breakpoint(width: number): 'mobile' | 'tablet' | 'desktop';
  AdaptiveShell(id: string, opts?: UiNodeBaseOptions & {
    width?: number; mode?: 'mobile' | 'tablet' | 'desktop'; title?: string;
    body?: UiNode | UiNode[]; appBar?: UiNode; bottomNav?: UiNode; sidebar?: UiNode; rail?: UiNode;
    nav?: Array<{ label: string; icon?: string; active?: boolean }>;
    tabs?: Array<{ icon?: string; active?: boolean }>;
    actions?: UiNode[]; back?: boolean; onBack?: (event: UiEvent) => void;
    sidebarWidth?: number;
  }): UiContainerHandle;
}

export interface SofuuUi {
  isAttached(): boolean;
  /** Theme of the active scene ("light" | "dark"); component kits use it to pick readable colors. */
  theme(): UiThemeMode;
  app(title?: string, opts?: { theme?: UiThemeMode; padding?: number; gap?: number }): UiAppBuilder;
  surface(id?: string, opts?: { title?: string; theme?: UiThemeMode }): UiView;

  row(children: UiNode[], opts?: UiContainerOptions): UiContainerHandle;
  row(id: string, children: UiNode[], opts?: UiContainerOptions): UiContainerHandle;
  col(children: UiNode[], opts?: UiContainerOptions): UiContainerHandle;
  col(id: string, children: UiNode[], opts?: UiContainerOptions): UiContainerHandle;
  grid(children: UiNode[], opts?: UiContainerOptions): UiContainerHandle;
  grid(id: string, children: UiNode[], opts?: UiContainerOptions): UiContainerHandle;

  tabs(items: Array<{ label: string; child: UiNode }>, opts?: UiTabsOptions): UiTabsHandle;
  tabs(id: string, items: Array<{ label: string; child: UiNode }>, opts?: UiTabsOptions): UiTabsHandle;

  text(content: string, opts?: UiTextOptions): UiTextHandle;
  text(id: string, content: string, opts?: UiTextOptions): UiTextHandle;
  title(content: string, opts?: UiTextOptions): UiTextHandle;

  rule(opts?: UiRuleOptions): UiNodeHandle;
  rule(id?: string, opts?: UiRuleOptions): UiNodeHandle;
  spacer(opts?: UiNodeBaseOptions): UiNodeHandle;
  spacer(id?: string, opts?: UiNodeBaseOptions): UiNodeHandle;
  themeToggle(opts?: UiThemeToggleOptions): UiNodeHandle;
  themeToggle(id?: string, opts?: UiThemeToggleOptions): UiNodeHandle;

  segmented(options: string[], opts?: UiSegmentedOptions): UiSegmentedHandle;
  segmented(options: string[], active?: number, opts?: UiSegmentedOptions): UiSegmentedHandle;
  segmented(
    id: string,
    options: string[],
    active?: number,
    opts?: UiSegmentedOptions,
  ): UiSegmentedHandle;

  button(label: string, opts?: UiButtonOptions): UiButtonHandle;
  button(id: string, label: string, opts?: UiButtonOptions): UiButtonHandle;

  switch(checked?: boolean, opts?: UiSwitchOptions): UiSwitchHandle;
  switch(checked: boolean, label?: string | null, opts?: UiSwitchOptions): UiSwitchHandle;
  switch(
    id: string,
    checked?: boolean,
    label?: string | null,
    opts?: UiSwitchOptions,
  ): UiSwitchHandle;

  input(placeholder?: string, opts?: UiInputOptions): UiInputHandle;
  input(id: string, placeholder?: string, opts?: UiInputOptions): UiInputHandle;

  richText(spans: Array<string | UiTextSpan>, opts?: UiRichTextOptions): UiRichTextHandle;
  richText(id: string, spans: Array<string | UiTextSpan>, opts?: UiRichTextOptions): UiRichTextHandle;

  icon(name: string, opts?: UiIconOptions): UiIconHandle;
  icon(id: string, name: string, opts?: UiIconOptions): UiIconHandle;

  image(src: string, opts?: UiImageOptions): UiImageHandle;
  image(id: string, src: string, opts?: UiImageOptions): UiImageHandle;

  stack(children: UiNode[], opts?: UiStackOptions): UiStackHandle;
  stack(id: string, children: UiNode[], opts?: UiStackOptions): UiStackHandle;

  safeArea(child: UiNode | null, opts?: UiSafeAreaOptions): UiNodeHandle;
  safeArea(id: string, child: UiNode | null, opts?: UiSafeAreaOptions): UiNodeHandle;

  overlay(child: UiNode | null, opts?: UiOverlayOptions): UiNodeHandle;
  overlay(id: string, child: UiNode | null, opts?: UiOverlayOptions): UiNodeHandle;

  navStack(routes: UiNavRoute[], opts?: UiNavStackOptions): UiNavStackHandle;
  navStack(id: string, routes: UiNavRoute[], opts?: UiNavStackOptions): UiNavStackHandle;

  listView(items: UiListItem[], opts?: UiListViewOptions): UiListViewHandle;
  listView(id: string, items: UiListItem[], opts?: UiListViewOptions): UiListViewHandle;

  textArea(placeholder?: string, opts?: UiTextAreaOptions): UiTextAreaHandle;
  textArea(id: string, placeholder?: string, opts?: UiTextAreaOptions): UiTextAreaHandle;

  slider(opts?: UiSliderOptions): UiSliderHandle;
  slider(id: string, opts?: UiSliderOptions): UiSliderHandle;

  rangeSlider(opts?: UiRangeSliderOptions): UiRangeSliderHandle;
  rangeSlider(id: string, opts?: UiRangeSliderOptions): UiRangeSliderHandle;

  progress(opts?: UiProgressOptions): UiProgressHandle;
  progress(id: string, opts?: UiProgressOptions): UiProgressHandle;

  pressable(child: UiNode | null, opts?: UiPressableOptions): UiNodeHandle;
  pressable(id: string, child: UiNode | null, opts?: UiPressableOptions): UiNodeHandle;

  draggable(child: UiNode | null, opts?: UiDraggableOptions): UiDraggableHandle;
  draggable(id: string, child: UiNode | null, opts?: UiDraggableOptions): UiDraggableHandle;

  stat(label: string, value: number, opts?: UiStatOptions): UiStatHandle;
  stat(id: string, label: string, value: number, opts?: UiStatOptions): UiStatHandle;

  pane(id: string, opts?: UiPlotPaneInput): UiPlotPane;

  plot(opts?: UiPlotOptions): UiPlotHandle;
  plot(id: string, opts?: UiPlotOptions): UiPlotHandle;

  table(opts?: UiTableOptions): UiTableHandle;
  table(id: string, opts?: UiTableOptions): UiTableHandle;

  feed(opts?: UiFeedOptions): UiFeedHandle;
  feed(id: string, opts?: UiFeedOptions): UiFeedHandle;

  graph(opts?: UiGraphOptions): UiGraphHandle;
  graph(id: string, opts?: UiGraphOptions): UiGraphHandle;

  code(content: string, opts?: UiCodeOptions): UiCodeHandle;
  code(id: string, content: string, opts?: UiCodeOptions): UiCodeHandle;

  /** Dependency-free observable cell (`get` / `set` / `update` / `subscribe`). */
  signal<T>(initial: T): UiSignal<T>;
  /** Derived cell, recomputed when any dependency changes. */
  computed<T>(compute: () => T, deps?: Array<UiSignal<unknown>>): UiSignal<T>;
  /** Run now and on every dependency change; the returned function disposes. */
  effect(run: () => void | (() => void), deps?: Array<UiSignal<unknown>>): () => void;

  enableAuto(enabled: boolean): void;
  onPrompt(handler: ((prompt: string, composer: UiAutoComposer) => void) | null): void;
  autoComposer(opts?: {
    title?: string;
    theme?: UiThemeMode;
    agentName?: string;
    profile?: string;
  }): UiAutoComposer;
}

export interface UiAutoComposer {
  readonly view: UiView;
  buildScene(): UiScene;
  mount(): UiAutoComposer;
  ingestEvent(event: Record<string, unknown>): void;
  updateTriggers(stats: Array<Record<string, unknown>>): void;
  updateStateRows(rows: UiTableRow[]): void;
  clearApproval(id: string): void;
}
