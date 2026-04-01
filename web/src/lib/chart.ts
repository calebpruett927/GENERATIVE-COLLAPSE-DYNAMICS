/**
 * Professional Chart Utilities for GCD Website
 *
 * Provides HiDPI-aware canvas setup, axis rendering, grid lines,
 * hover crosshairs, gradient fills, and legend drawing.
 * Used by all interactive chart components.
 */

/* ─── Types ─────────────────────────────────────────────────────── */

export interface ChartMargins {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface AxisConfig {
  label: string;
  min: number;
  max: number;
  ticks?: number;        // number of tick marks (default 5)
  format?: (v: number) => string;
}

export interface ChartConfig {
  margins?: Partial<ChartMargins>;
  bgColor?: string;
  gridColor?: string;
  gridAlpha?: number;
  axisColor?: string;
  fontFamily?: string;
  fontSize?: number;
  labelFontSize?: number;
  tickLength?: number;
}

export interface ChartArea {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  dpr: number;
  width: number;        // CSS width
  height: number;       // CSS height
  margins: ChartMargins;
  plotX: number;         // plot area left edge (CSS px)
  plotY: number;         // plot area top edge (CSS px)
  plotW: number;         // plot area width (CSS px)
  plotH: number;         // plot area height (CSS px)
}

/* ─── Constants ─────────────────────────────────────────────────── */

export const CHART_COLORS = {
  bg: '#0c0c14',
  grid: '#1e293b',
  axis: '#64748b',
  text: '#94a3b8',
  textDim: '#475569',
  amber: '#f59e0b',
  blue: '#3b82f6',
  green: '#34d399',
  red: '#ef4444',
  purple: '#a855f7',
  cyan: '#22d3ee',
  white: '#e2e8f0',
  stableGreen: '#059669',
  watchAmber: '#d97706',
  collapseRed: '#dc2626',
};

const DEFAULT_MARGINS: ChartMargins = { top: 24, right: 20, bottom: 44, left: 56 };
const DEFAULT_CONFIG: Required<ChartConfig> = {
  margins: DEFAULT_MARGINS,
  bgColor: CHART_COLORS.bg,
  gridColor: CHART_COLORS.grid,
  gridAlpha: 0.6,
  axisColor: CHART_COLORS.axis,
  fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
  fontSize: 11,
  labelFontSize: 12,
  tickLength: 5,
};

/* ─── Setup ─────────────────────────────────────────────────────── */

/**
 * Initialize a canvas for HiDPI rendering. Sets canvas buffer size
 * to devicePixelRatio × CSS size for crisp rendering on Retina displays.
 */
export function setupCanvas(
  canvas: HTMLCanvasElement,
  cssWidth: number,
  cssHeight: number,
  config?: ChartConfig,
): ChartArea {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const margins: ChartMargins = { ...DEFAULT_MARGINS, ...(cfg.margins ?? {}) };
  const dpr = Math.min(window.devicePixelRatio || 1, 3);

  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;

  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);

  // Enable sub-pixel anti-aliasing
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  const plotX = margins.left;
  const plotY = margins.top;
  const plotW = cssWidth - margins.left - margins.right;
  const plotH = cssHeight - margins.top - margins.bottom;

  return { canvas, ctx, dpr, width: cssWidth, height: cssHeight, margins, plotX, plotY, plotW, plotH };
}

/**
 * Clear the canvas with the background color and draw the plot area border.
 */
export function clearChart(area: ChartArea, config?: ChartConfig): void {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const { ctx, width, height, plotX, plotY, plotW, plotH } = area;

  ctx.clearRect(0, 0, width, height);

  // Background
  ctx.fillStyle = cfg.bgColor;
  ctx.fillRect(0, 0, width, height);

  // Plot area subtle background
  ctx.fillStyle = 'rgba(15, 23, 42, 0.4)';
  ctx.fillRect(plotX, plotY, plotW, plotH);
}

/* ─── Grid & Axes ──────────────────────────────────────────────── */

/**
 * Draw grid lines inside the plot area.
 */
export function drawGrid(
  area: ChartArea,
  xAxis: AxisConfig,
  yAxis: AxisConfig,
  config?: ChartConfig,
): void {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const { ctx, plotX, plotY, plotW, plotH } = area;
  const xTicks = xAxis.ticks ?? 5;
  const yTicks = yAxis.ticks ?? 5;

  ctx.save();
  ctx.strokeStyle = cfg.gridColor;
  ctx.lineWidth = 0.75;
  ctx.globalAlpha = cfg.gridAlpha;

  // Vertical grid lines
  for (let i = 0; i <= xTicks; i++) {
    const x = plotX + (i / xTicks) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, plotY);
    ctx.lineTo(x, plotY + plotH);
    ctx.stroke();
  }

  // Horizontal grid lines
  for (let i = 0; i <= yTicks; i++) {
    const y = plotY + (i / yTicks) * plotH;
    ctx.beginPath();
    ctx.moveTo(plotX, y);
    ctx.lineTo(plotX + plotW, y);
    ctx.stroke();
  }

  ctx.restore();
}

/**
 * Draw X and Y axes with tick marks and labels.
 */
export function drawAxes(
  area: ChartArea,
  xAxis: AxisConfig,
  yAxis: AxisConfig,
  config?: ChartConfig,
): void {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const { ctx, plotX, plotY, plotW, plotH } = area;
  const xTicks = xAxis.ticks ?? 5;
  const yTicks = yAxis.ticks ?? 5;
  const formatX = xAxis.format ?? ((v: number) => v % 1 === 0 ? v.toString() : v.toFixed(2));
  const formatY = yAxis.format ?? ((v: number) => v % 1 === 0 ? v.toString() : v.toFixed(2));

  ctx.save();

  // Axis lines
  ctx.strokeStyle = cfg.axisColor;
  ctx.lineWidth = 1;

  // X-axis (bottom)
  ctx.beginPath();
  ctx.moveTo(plotX, plotY + plotH);
  ctx.lineTo(plotX + plotW, plotY + plotH);
  ctx.stroke();

  // Y-axis (left)
  ctx.beginPath();
  ctx.moveTo(plotX, plotY);
  ctx.lineTo(plotX, plotY + plotH);
  ctx.stroke();

  // Tick marks and labels
  ctx.fillStyle = cfg.axisColor;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.font = `${cfg.fontSize}px ${cfg.fontFamily}`;

  // X-axis ticks
  for (let i = 0; i <= xTicks; i++) {
    const x = plotX + (i / xTicks) * plotW;
    const val = xAxis.min + (i / xTicks) * (xAxis.max - xAxis.min);

    ctx.beginPath();
    ctx.moveTo(x, plotY + plotH);
    ctx.lineTo(x, plotY + plotH + cfg.tickLength);
    ctx.stroke();

    ctx.fillText(formatX(val), x, plotY + plotH + cfg.tickLength + 3);
  }

  // Y-axis ticks
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= yTicks; i++) {
    const y = plotY + plotH - (i / yTicks) * plotH;
    const val = yAxis.min + (i / yTicks) * (yAxis.max - yAxis.min);

    ctx.beginPath();
    ctx.moveTo(plotX - cfg.tickLength, y);
    ctx.lineTo(plotX, y);
    ctx.stroke();

    ctx.fillText(formatY(val), plotX - cfg.tickLength - 3, y);
  }

  // Axis labels
  ctx.fillStyle = CHART_COLORS.text;
  ctx.font = `${cfg.labelFontSize}px ${cfg.fontFamily}`;

  // X-axis label
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(xAxis.label, plotX + plotW / 2, plotY + plotH + 28);

  // Y-axis label (rotated)
  ctx.save();
  ctx.translate(14, plotY + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(yAxis.label, 0, 0);
  ctx.restore();

  ctx.restore();
}

/* ─── Plotting Helpers ─────────────────────────────────────────── */

/** Map a data value to pixel X inside the plot area. */
export function mapX(area: ChartArea, value: number, min: number, max: number): number {
  return area.plotX + ((value - min) / (max - min)) * area.plotW;
}

/** Map a data value to pixel Y inside the plot area (Y increases downward). */
export function mapY(area: ChartArea, value: number, min: number, max: number): number {
  return area.plotY + area.plotH - ((value - min) / (max - min)) * area.plotH;
}

/**
 * Draw a line curve from data points.
 */
export function drawCurve(
  area: ChartArea,
  data: { x: number; y: number }[],
  xRange: { min: number; max: number },
  yRange: { min: number; max: number },
  options: {
    color: string;
    lineWidth?: number;
    dash?: number[];
    fill?: string | CanvasGradient;
  },
): void {
  if (data.length < 2) return;
  const { ctx, plotX, plotY, plotW, plotH } = area;

  ctx.save();
  ctx.beginPath();
  // Clip to plot area
  ctx.rect(plotX, plotY, plotW, plotH);
  ctx.clip();

  ctx.strokeStyle = options.color;
  ctx.lineWidth = options.lineWidth ?? 2;
  if (options.dash) ctx.setLineDash(options.dash);

  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const px = mapX(area, data[i].x, xRange.min, xRange.max);
    const py = mapY(area, data[i].y, yRange.min, yRange.max);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Optional fill under curve
  if (options.fill) {
    const last = data[data.length - 1];
    const first = data[0];
    ctx.lineTo(mapX(area, last.x, xRange.min, xRange.max), plotY + plotH);
    ctx.lineTo(mapX(area, first.x, xRange.min, xRange.max), plotY + plotH);
    ctx.closePath();
    ctx.fillStyle = options.fill;
    ctx.fill();
  }

  ctx.restore();
}

/**
 * Draw a vertical marker line with optional label.
 */
export function drawVerticalMarker(
  area: ChartArea,
  xValue: number,
  xRange: { min: number; max: number },
  options: {
    color: string;
    dash?: number[];
    label?: string;
    labelPosition?: 'top' | 'bottom';
    lineWidth?: number;
  },
): void {
  const { ctx, plotY, plotH } = area;
  const px = mapX(area, xValue, xRange.min, xRange.max);

  ctx.save();
  ctx.strokeStyle = options.color;
  ctx.lineWidth = options.lineWidth ?? 1;
  if (options.dash) ctx.setLineDash(options.dash);

  ctx.beginPath();
  ctx.moveTo(px, plotY);
  ctx.lineTo(px, plotY + plotH);
  ctx.stroke();

  if (options.label) {
    ctx.setLineDash([]);
    ctx.fillStyle = options.color;
    ctx.font = `10px ${DEFAULT_CONFIG.fontFamily}`;
    ctx.textAlign = 'center';
    if (options.labelPosition === 'bottom') {
      ctx.textBaseline = 'bottom';
      ctx.fillText(options.label, px, plotY + plotH - 4);
    } else {
      ctx.textBaseline = 'top';
      ctx.fillText(options.label, px, plotY + 4);
    }
  }

  ctx.restore();
}

/**
 * Draw a horizontal marker line.
 */
export function drawHorizontalMarker(
  area: ChartArea,
  yValue: number,
  yRange: { min: number; max: number },
  options: {
    color: string;
    dash?: number[];
    label?: string;
    lineWidth?: number;
  },
): void {
  const { ctx, plotX, plotW } = area;
  const py = mapY(area, yValue, yRange.min, yRange.max);

  ctx.save();
  ctx.strokeStyle = options.color;
  ctx.lineWidth = options.lineWidth ?? 1;
  if (options.dash) ctx.setLineDash(options.dash);

  ctx.beginPath();
  ctx.moveTo(plotX, py);
  ctx.lineTo(plotX + plotW, py);
  ctx.stroke();

  if (options.label) {
    ctx.setLineDash([]);
    ctx.fillStyle = options.color;
    ctx.font = `10px ${DEFAULT_CONFIG.fontFamily}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(options.label, plotX + 4, py - 3);
  }

  ctx.restore();
}

/**
 * Draw a point marker (filled circle with optional outline).
 */
export function drawPoint(
  area: ChartArea,
  xValue: number,
  yValue: number,
  xRange: { min: number; max: number },
  yRange: { min: number; max: number },
  options: {
    color: string;
    radius?: number;
    strokeColor?: string;
    strokeWidth?: number;
    glow?: boolean;
  },
): void {
  const { ctx } = area;
  const px = mapX(area, xValue, xRange.min, xRange.max);
  const py = mapY(area, yValue, yRange.min, yRange.max);
  const r = options.radius ?? 4;

  ctx.save();

  if (options.glow) {
    ctx.shadowColor = options.color;
    ctx.shadowBlur = 8;
  }

  ctx.fillStyle = options.color;
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fill();

  if (options.strokeColor) {
    ctx.strokeStyle = options.strokeColor;
    ctx.lineWidth = options.strokeWidth ?? 1.5;
    ctx.stroke();
  }

  ctx.restore();
}

/* ─── Legend ────────────────────────────────────────────────────── */

export interface LegendItem {
  label: string;
  color: string;
  dash?: number[];
}

/**
 * Draw a legend box inside the chart area.
 */
export function drawLegend(
  area: ChartArea,
  items: LegendItem[],
  position: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' = 'top-right',
): void {
  const { ctx, plotX, plotY, plotW } = area;
  const padding = 8;
  const lineH = 18;
  const swatchW = 20;
  const font = `11px ${DEFAULT_CONFIG.fontFamily}`;

  ctx.save();
  ctx.font = font;

  // Measure width
  let maxTextW = 0;
  for (const item of items) {
    const w = ctx.measureText(item.label).width;
    if (w > maxTextW) maxTextW = w;
  }

  const boxW = swatchW + 8 + maxTextW + padding * 2;
  const boxH = items.length * lineH + padding * 2;

  let bx: number, by: number;
  if (position === 'top-right') { bx = plotX + plotW - boxW - 8; by = plotY + 8; }
  else if (position === 'top-left') { bx = plotX + 8; by = plotY + 8; }
  else if (position === 'bottom-right') { bx = plotX + plotW - boxW - 8; by = plotY + area.plotH - boxH - 8; }
  else { bx = plotX + 8; by = plotY + area.plotH - boxH - 8; }

  // Background
  ctx.fillStyle = 'rgba(10, 10, 20, 0.85)';
  ctx.strokeStyle = 'rgba(100, 116, 139, 0.3)';
  ctx.lineWidth = 1;
  roundRect(ctx, bx, by, boxW, boxH, 6);
  ctx.fill();
  ctx.stroke();

  // Items
  for (let i = 0; i < items.length; i++) {
    const iy = by + padding + i * lineH + lineH / 2;
    const ix = bx + padding;

    // Swatch line
    ctx.strokeStyle = items[i].color;
    ctx.lineWidth = 2.5;
    if (items[i].dash) ctx.setLineDash(items[i].dash!);
    else ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(ix, iy);
    ctx.lineTo(ix + swatchW, iy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label
    ctx.fillStyle = CHART_COLORS.text;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(items[i].label, ix + swatchW + 8, iy);
  }

  ctx.restore();
}

/* ─── Crosshair / Hover ────────────────────────────────────────── */

/**
 * Draw hover crosshairs at a data point with a tooltip.
 */
export function drawCrosshair(
  area: ChartArea,
  xValue: number,
  yValue: number,
  xRange: { min: number; max: number },
  yRange: { min: number; max: number },
  tooltip?: string,
): void {
  const { ctx, plotX, plotY, plotW, plotH } = area;
  const px = mapX(area, xValue, xRange.min, xRange.max);
  const py = mapY(area, yValue, yRange.min, yRange.max);

  ctx.save();

  // Crosshair lines
  ctx.strokeStyle = 'rgba(226, 232, 240, 0.25)';
  ctx.lineWidth = 0.75;
  ctx.setLineDash([3, 3]);

  ctx.beginPath();
  ctx.moveTo(px, plotY);
  ctx.lineTo(px, plotY + plotH);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(plotX, py);
  ctx.lineTo(plotX + plotW, py);
  ctx.stroke();
  ctx.setLineDash([]);

  // Center dot
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(px, py, 3, 0, Math.PI * 2);
  ctx.fill();

  // Tooltip
  if (tooltip) {
    ctx.font = `11px ${DEFAULT_CONFIG.fontFamily}`;
    const tw = ctx.measureText(tooltip).width;
    const tp = 6;
    let tx = px + 10;
    let ty = py - 24;
    if (tx + tw + tp * 2 > plotX + plotW) tx = px - tw - tp * 2 - 10;
    if (ty < plotY) ty = py + 10;

    ctx.fillStyle = 'rgba(10, 10, 20, 0.92)';
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.3)';
    ctx.lineWidth = 1;
    roundRect(ctx, tx, ty, tw + tp * 2, 22, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = CHART_COLORS.white;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(tooltip, tx + tp, ty + 11);
  }

  ctx.restore();
}

/* ─── Heatmap Helpers ──────────────────────────────────────────── */

/** Color interpolation for heatmaps. Returns [r, g, b]. */
export function heatmapColor(
  value: number,
  scheme: 'viridis' | 'inferno' | 'plasma' | 'coolwarm' | 'gcd' = 'gcd',
): [number, number, number] {
  const t = Math.max(0, Math.min(1, value));

  if (scheme === 'gcd') {
    // Custom GCD palette: deep blue → teal → amber → white
    if (t < 0.25) {
      const s = t / 0.25;
      return [
        Math.floor(8 + 2 * s),
        Math.floor(12 + 48 * s),
        Math.floor(60 + 40 * s),
      ];
    } else if (t < 0.5) {
      const s = (t - 0.25) / 0.25;
      return [
        Math.floor(10 + 20 * s),
        Math.floor(60 + 80 * s),
        Math.floor(100 - 30 * s),
      ];
    } else if (t < 0.75) {
      const s = (t - 0.5) / 0.25;
      return [
        Math.floor(30 + 190 * s),
        Math.floor(140 + 50 * s),
        Math.floor(70 - 40 * s),
      ];
    } else {
      const s = (t - 0.75) / 0.25;
      return [
        Math.floor(220 + 35 * s),
        Math.floor(190 + 55 * s),
        Math.floor(30 + 160 * s),
      ];
    }
  }

  if (scheme === 'viridis') {
    if (t < 0.33) {
      const s = t / 0.33;
      return [Math.floor(68 * (1 - s) + 49 * s), Math.floor(1 + 104 * s), Math.floor(84 + 65 * s)];
    } else if (t < 0.66) {
      const s = (t - 0.33) / 0.33;
      return [Math.floor(49 + 81 * s), Math.floor(105 + 80 * s), Math.floor(149 - 70 * s)];
    } else {
      const s = (t - 0.66) / 0.34;
      return [Math.floor(130 + 123 * s), Math.floor(185 + 50 * s), Math.floor(79 - 60 * s)];
    }
  }

  if (scheme === 'inferno') {
    if (t < 0.33) {
      const s = t / 0.33;
      return [Math.floor(3 + 97 * s), Math.floor(5 * (1 - s * 0.5)), Math.floor(20 + 60 * s)];
    } else if (t < 0.66) {
      const s = (t - 0.33) / 0.33;
      return [Math.floor(100 + 120 * s), Math.floor(3 + 60 * s), Math.floor(80 - 50 * s)];
    } else {
      const s = (t - 0.66) / 0.34;
      return [Math.floor(220 + 32 * s), Math.floor(63 + 170 * s), Math.floor(30 + 30 * s)];
    }
  }

  if (scheme === 'coolwarm') {
    if (t < 0.5) {
      const s = t / 0.5;
      return [
        Math.floor(59 + 180 * s),
        Math.floor(76 + 155 * s),
        Math.floor(192 + 50 * s),
      ];
    } else {
      const s = (t - 0.5) / 0.5;
      return [
        Math.floor(239 - 18 * s),
        Math.floor(231 - 170 * s),
        Math.floor(242 - 195 * s),
      ];
    }
  }

  // plasma
  if (t < 0.33) {
    const s = t / 0.33;
    return [Math.floor(13 + 135 * s), Math.floor(8 + 5 * s), Math.floor(135 + 20 * s)];
  } else if (t < 0.66) {
    const s = (t - 0.33) / 0.33;
    return [Math.floor(148 + 80 * s), Math.floor(13 + 70 * s), Math.floor(155 - 80 * s)];
  } else {
    const s = (t - 0.66) / 0.34;
    return [Math.floor(228 + 12 * s), Math.floor(83 + 130 * s), Math.floor(75 - 55 * s)];
  }
}

/**
 * Render a heatmap from a 2D value grid into a canvas with HiDPI support.
 * Returns the ImageData for overlay drawing.
 */
export function renderHeatmap(
  area: ChartArea,
  values: number[][],
  options: {
    scheme?: 'viridis' | 'inferno' | 'plasma' | 'coolwarm' | 'gcd';
    normalizeMax?: number;
  } = {},
): void {
  const { ctx, plotX, plotY, plotW, plotH } = area;
  const rows = values.length;
  const cols = values[0]?.length ?? 0;
  if (rows === 0 || cols === 0) return;

  const scheme = options.scheme ?? 'gcd';

  // Find max value for normalization
  let maxVal = options.normalizeMax ?? 0;
  if (maxVal <= 0) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = values[r][c];
        if (isFinite(v) && v > maxVal) maxVal = v;
      }
    }
  }
  if (maxVal <= 0) maxVal = 1;

  // Create heatmap image at native resolution
  const imgW = Math.max(cols, Math.round(plotW));
  const imgH = Math.max(rows, Math.round(plotH));
  const offscreen = new OffscreenCanvas(cols, rows);
  const offCtx = offscreen.getContext('2d')!;
  const imageData = offCtx.createImageData(cols, rows);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = values[r][c];
      const norm = isFinite(v) ? Math.min(v / maxVal, 1.0) : 0;
      const [cr, cg, cb] = heatmapColor(norm, scheme);
      const idx = (r * cols + c) * 4;
      imageData.data[idx] = cr;
      imageData.data[idx + 1] = cg;
      imageData.data[idx + 2] = cb;
      imageData.data[idx + 3] = 255;
    }
  }

  offCtx.putImageData(imageData, 0, 0);

  // Draw the heatmap scaled to the plot area
  ctx.save();
  ctx.imageSmoothingEnabled = false; // keep pixel-sharp for heatmaps
  ctx.drawImage(offscreen, plotX, plotY, plotW, plotH);
  ctx.restore();
}

/* ─── Color Bar (Legend for Heatmaps) ──────────────────────────── */

/**
 * Draw a vertical color bar legend next to the plot area.
 */
export function drawColorBar(
  area: ChartArea,
  minLabel: string,
  maxLabel: string,
  scheme: 'viridis' | 'inferno' | 'plasma' | 'coolwarm' | 'gcd' = 'gcd',
  midLabel?: string,
): void {
  const { ctx, plotY, plotH, plotX, plotW } = area;
  const barX = plotX + plotW + 8;
  const barW = 14;
  const barY = plotY;
  const barH = plotH;

  ctx.save();

  // Draw gradient bar
  for (let i = 0; i < barH; i++) {
    const t = 1 - i / barH; // top = high, bottom = low
    const [r, g, b] = heatmapColor(t, scheme);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(barX, barY + i, barW, 1);
  }

  // Border
  ctx.strokeStyle = CHART_COLORS.axis;
  ctx.lineWidth = 0.5;
  ctx.strokeRect(barX, barY, barW, barH);

  // Labels
  ctx.fillStyle = CHART_COLORS.text;
  ctx.font = `10px ${DEFAULT_CONFIG.fontFamily}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(maxLabel, barX + barW + 4, barY);
  ctx.textBaseline = 'bottom';
  ctx.fillText(minLabel, barX + barW + 4, barY + barH);

  if (midLabel) {
    ctx.textBaseline = 'middle';
    ctx.fillText(midLabel, barX + barW + 4, barY + barH / 2);
  }

  ctx.restore();
}

/* ─── Utility ──────────────────────────────────────────────────── */

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

/**
 * Get mouse position in CSS pixels relative to canvas.
 */
export function getMousePos(
  canvas: HTMLCanvasElement,
  event: MouseEvent,
): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

/**
 * Convert mouse CSS position to data coordinates.
 */
export function mouseToData(
  area: ChartArea,
  mouseX: number,
  mouseY: number,
  xRange: { min: number; max: number },
  yRange: { min: number; max: number },
): { x: number; y: number; inPlot: boolean } {
  const { plotX, plotY, plotW, plotH } = area;
  const inPlot = mouseX >= plotX && mouseX <= plotX + plotW &&
                 mouseY >= plotY && mouseY <= plotY + plotH;
  const x = xRange.min + ((mouseX - plotX) / plotW) * (xRange.max - xRange.min);
  const y = yRange.max - ((mouseY - plotY) / plotH) * (yRange.max - yRange.min);
  return { x, y, inPlot };
}
