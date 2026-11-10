package com.sofuu.ui.soapbubble

import android.content.Context
import android.database.ContentObserver
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Icon
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.setProgress
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpOffset
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.sofuu.ui.components.Icons2
import com.sofuu.ui.theme.LocalSofuuColors
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * The six colors the bubble components read from the host theme. Sofuu's
 * token set (ui/theme/Color.kt) maps onto the reference web palette:
 * text→foreground, bgCard→card, bgCardHover→cardShade, textMuted→muted,
 * accent→accentDeep, dark→isDark.
 */
private data class BubbleColors(
    val foreground: Color,
    val card: Color,
    val cardShade: Color,
    val muted: Color,
    val accentDeep: Color,
    val isDark: Boolean,
)

/** Shorthand: SofuuBubble.colors.foreground instead of a CompositionLocal read. */
private object SofuuBubble {
    val colors: BubbleColors
        @Composable get() = LocalSofuuColors.current.let {
            BubbleColors(
                foreground = it.text,
                card = it.bgCard,
                cardShade = it.bgCardHover,
                muted = it.textMuted,
                accentDeep = it.accent,
                isDark = it.dark,
            )
        }
}

val LocalSoapBubbleSettings = staticCompositionLocalOf { SoapBubbleSettings.lightPreset }

/**
 * The shared material. Sofuu has light AND dark themes, so the host picks the
 * preset once at the root (light glass preset / dark rainbow defaults) and
 * every control below inherits it and the same shared backdrop.
 */
@Composable
fun SoapBubbleMaterial(
    settings: SoapBubbleSettings? = null,
    content: @Composable () -> Unit,
) {
    val base = settings ?: SoapBubbleSettings.lightPreset
    val lifecycle = LocalLifecycleOwner.current
    DisposableEffect(lifecycle) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> SoapBubbleGl.setAppActive(true)
                Lifecycle.Event.ON_STOP -> SoapBubbleGl.setAppActive(false)
                else -> Unit
            }
        }
        lifecycle.lifecycle.addObserver(observer)
        onDispose {
            lifecycle.lifecycle.removeObserver(observer)
            SoapBubbleGl.setAppActive(true)
        }
    }
    CompositionLocalProvider(
        LocalSoapBubbleSettings provides base.normalized(),
        content = content,
    )
}

@Composable
fun rememberReducedMotion(): Boolean {
    val context = LocalContext.current
    var reduced by remember { mutableStateOf(readReducedMotion(context)) }
    DisposableEffect(context) {
        val observer = object : ContentObserver(Handler(Looper.getMainLooper())) {
            override fun onChange(selfChange: Boolean) {
                reduced = readReducedMotion(context)
            }
        }
        context.contentResolver.registerContentObserver(
            Settings.Global.getUriFor(Settings.Global.ANIMATOR_DURATION_SCALE),
            false,
            observer,
        )
        onDispose { context.contentResolver.unregisterContentObserver(observer) }
    }
    return reduced
}

private fun readReducedMotion(context: Context): Boolean = runCatching {
    Settings.Global.getFloat(
        context.contentResolver,
        Settings.Global.ANIMATOR_DURATION_SCALE,
        1f,
    ) == 0f
}.getOrDefault(false)

/* ------------------------------------------------------------------ */
/* Surface                                                             */
/* ------------------------------------------------------------------ */

/**
 * A bubble panel or button. The film is drawn behind [content]; the content
 * keeps its own size while the film wobbles and swells around it.
 */
@Composable
fun SoapBubbleSurface(
    modifier: Modifier = Modifier,
    settings: SoapBubbleSettings = LocalSoapBubbleSettings.current,
    radius: Float = 24f,
    bevel: Float = settings.bevel,
    interactive: Boolean = true,
    enabled: Boolean = true,
    onClick: (() -> Unit)? = null,
    role: Role? = null,
    fontSize: TextUnit = 16.sp,
    shadow: Boolean = false,
    fallback: Boolean = true,
    hostSize: DpSize? = null,
    canvasSize: DpSize? = null,
    center: DpOffset? = null,
    layer: SoapBubbleLayer? = null,
    contentPadding: PaddingValues = PaddingValues(0.dp),
    content: @Composable BoxScope.() -> Unit,
) {
    val colors = SofuuBubble.colors
    val density = LocalDensity.current
    val view = LocalView.current
    val reduced = rememberReducedMotion()
    val controller = remember { SoapBubbleController(SoapBubbleIds.newId()) }
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val fontPx = with(density) { fontSize.toPx() }
    val bleedPx = max(with(density) { 12.dp.toPx() }, fontPx)

    controller.settings = settings.copy(radius = radius, bevel = bevel)
    controller.density = density.density
    controller.fontPx = fontPx
    controller.bleedPx = bleedPx
    controller.reducedMotion = reduced
    controller.isButton = onClick != null
    controller.interactive = interactive
    controller.visible = true

    DisposableEffect(controller) {
        SoapBubbleGl.register(controller)
        onDispose { SoapBubbleGl.unregister(controller) }
    }
    LaunchedEffect(pressed) { controller.press(pressed) }

    val clickModifier = if (onClick != null) {
        Modifier.clickable(
            enabled = enabled,
            role = role,
            interactionSource = interaction,
            indication = null,
            onClick = onClick,
        )
    } else Modifier

    val pointerModifier = if (interactive) {
        Modifier.pointerInput(controller, enabled) {
            awaitPointerEventScope {
                while (true) {
                    val event = awaitPointerEvent(PointerEventPass.Initial)
                    val position = event.changes.firstOrNull()?.position ?: continue
                    val canvasW = if (controller.canvasWidthPx > 0f) controller.canvasWidthPx
                    else size.width.toFloat()
                    val canvasH = if (controller.canvasHeightPx > 0f) controller.canvasHeightPx
                    else size.height.toFloat()
                    val hostW = if (controller.hostWidthPx > 0f) controller.hostWidthPx
                    else size.width.toFloat()
                    val hostH = if (controller.hostHeightPx > 0f) controller.hostHeightPx
                    else size.height.toFloat()
                    val hostCenterX = controller.centerXPx - (canvasW - size.width) / 2f
                    val hostCenterY = controller.centerYPx - (canvasH - size.height) / 2f
                    val r = max(1f, min(hostW, hostH) / 2f)
                    val x = (position.x - hostCenterX) / r
                    val y = -(position.y - hostCenterY) / r
                    when (event.type) {
                        PointerEventType.Move,
                        PointerEventType.Enter,
                        -> controller.pointer(x, y, true)
                        PointerEventType.Exit -> controller.pointer(x, y, false)
                        PointerEventType.Press -> controller.pointer(x, y, true)
                        PointerEventType.Release -> controller.pointer(x, y, false)
                    }
                }
            }
        }
    } else Modifier

    Box(
        modifier
            .then(if (shadow) Modifier else Modifier)
            .onGloballyPositioned { coords ->
                val layoutW = coords.size.width.toFloat()
                val layoutH = coords.size.height.toFloat()
                val canvasWd = canvasSize?.width ?: (layoutW / density.density).dp.plus(0.dp)
                val canvasHd = canvasSize?.height ?: (layoutH / density.density).dp.plus(0.dp)
                val canvasWpx = if (canvasSize != null) canvasWd.value * density.density
                else layoutW + bleedPx * 2f
                val canvasHpx = if (canvasSize != null) canvasHd.value * density.density
                else layoutH + bleedPx * 2f
                val dx = (canvasWpx - layoutW) / 2f
                val dy = (canvasHpx - layoutH) / 2f
                controller.widthPx = layoutW
                controller.heightPx = layoutH
                controller.canvasWidthPx = canvasWpx
                controller.canvasHeightPx = canvasHpx
                controller.hostWidthPx =
                    (hostSize?.width?.value ?: (layoutW / density.density)) * density.density
                controller.hostHeightPx =
                    (hostSize?.height?.value ?: (layoutH / density.density)) * density.density
                controller.centerXPx = if (center != null) center.x.value * density.density + dx
                else canvasWpx / 2f
                controller.centerYPx = if (center != null) center.y.value * density.density + dy
                else canvasHpx / 2f
                controller.layer = layer?.let { it.copy(x = it.x + dx, y = it.y + dy) }
                val bounds = coords.boundsInWindow()
                controller.visible = bounds.width > 0f && bounds.height > 0f &&
                    bounds.bottom > 0f && bounds.top < view.height
                // A return from off-screen has to restart the shared loop.
                SoapBubbleGl.invalidate()
            }
            .then(clickModifier)
            .then(pointerModifier)
            .drawBehind {
                drawSoapBubble(
                    controller = controller,
                    radiusDp = radius,
                    colors = colors,
                    fallback = fallback,
                    layoutWidth = size.width,
                    layoutHeight = size.height,
                )
            }
            .padding(contentPadding),
        content = content,
    )
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawSoapBubble(
    controller: SoapBubbleController,
    radiusDp: Float,
    colors: BubbleColors,
    fallback: Boolean,
    layoutWidth: Float,
    layoutHeight: Float,
) {
    val frame = SoapBubbleGl.frame.value
    val region = frame?.regions?.get(controller.id)
    if (frame != null && region != null) {
        val canvasW = controller.canvasWidthPx.takeIf { it > 0f }
            ?: (layoutWidth + controller.bleedPx * 2f)
        val canvasH = controller.canvasHeightPx.takeIf { it > 0f }
            ?: (layoutHeight + controller.bleedPx * 2f)
        val dstOffset = IntOffset(
            ((layoutWidth - canvasW) / 2f).roundToInt(),
            ((layoutHeight - canvasH) / 2f).roundToInt(),
        )
        val dstSize = IntSize(canvasW.roundToInt(), canvasH.roundToInt())
        drawImage(
            image = frame.image,
            srcOffset = IntOffset(region.x, region.y),
            srcSize = IntSize(region.width, region.height),
            dstOffset = dstOffset,
            dstSize = dstSize,
            filterQuality = FilterQuality.Low,
        )
        return
    }
    if (!fallback) return

    // CSS fallback: the frosted palette the component shows before WebGL is
    // ready (or on a device without ES3).
    val w = size.width
    val h = size.height
    val corner = CornerRadius(min(radiusDp.dp.toPx(), min(w, h) / 2f))
    drawRoundRect(
        brush = Brush.verticalGradient(listOf(colors.card, colors.cardShade)),
        size = Size(w, h),
        cornerRadius = corner,
    )
    drawRoundRect(
        brush = Brush.linearGradient(
            colors = listOf(
                if (colors.isDark) Color.White.copy(alpha = 0.12f) else Color.White.copy(alpha = 0.42f),
                Color.Transparent,
            ),
            start = Offset.Zero,
            end = Offset(w, h),
        ),
        size = Size(w, h),
        cornerRadius = corner,
    )
    drawRoundRect(
        color = if (colors.isDark) Color.White.copy(alpha = 0.08f) else Color.White.copy(alpha = 0.28f),
        size = Size(w, h),
        cornerRadius = corner,
    )
}

/* ------------------------------------------------------------------ */
/* Button + icon                                                       */
/* ------------------------------------------------------------------ */

@Composable
fun SoapBubbleButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    settings: SoapBubbleSettings = LocalSoapBubbleSettings.current,
    contentColor: Color = SofuuBubble.colors.foreground,
    fontSize: TextUnit = 16.sp,
    minHeight: Dp = 44.dp,
    fillWidth: Boolean = true,
    radius: Float = 9999f,
    contentPadding: PaddingValues = PaddingValues(horizontal = 20.dp),
    content: @Composable RowScope.() -> Unit,
) {
    SoapBubbleSurface(
        modifier = modifier.heightIn(min = minHeight),
        settings = settings,
        radius = radius,
        interactive = true,
        enabled = enabled,
        onClick = onClick,
        role = Role.Button,
        fontSize = fontSize,
    ) {
        CompositionLocalProvider(LocalContentColor provides contentColor) {
            Row(
                Modifier
                    .defaultMinSize(minHeight = minHeight)
                    .padding(contentPadding)
                    .then(if (fillWidth) Modifier.fillMaxWidth() else Modifier),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically,
                content = content,
            )
        }
    }
}

/** CSS `#rrggbbaa`, the form the shader's tint expects. */
internal fun Color.toCssRgb(): String {
    val argb = toArgb()
    return String.format(
        "#%02x%02x%02x%02x",
        android.graphics.Color.red(argb),
        android.graphics.Color.green(argb),
        android.graphics.Color.blue(argb),
        android.graphics.Color.alpha(argb),
    )
}

@Composable
fun SoapBubbleIconButton(
    icon: ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    settings: SoapBubbleSettings = LocalSoapBubbleSettings.current,
    contentColor: Color = SofuuBubble.colors.foreground,
    iconSize: Dp = 20.dp,
) {
    SoapBubbleSurface(
        modifier = modifier.size(44.dp),
        settings = settings,
        radius = 9999f,
        interactive = true,
        enabled = enabled,
        onClick = onClick,
        role = Role.Button,
    ) {
        Box(Modifier.fillMaxWidth().height(44.dp), contentAlignment = Alignment.Center) {
            Icon(
                icon,
                contentDescription = contentDescription,
                tint = contentColor,
                modifier = Modifier.size(iconSize),
            )
        }
    }
}

/* ------------------------------------------------------------------ */
/* Toggle + slider                                                     */
/* ------------------------------------------------------------------ */

/**
 * The track is flat and the thumb is a real bubble lens, swelling while
 * pressed or moving, as the web switch's thumb does.
 */
@Composable
fun SoapBubbleToggle(
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    color: Color = Color(0xFF34C759),
    scale: Float = 1f,
    contentDescription: String? = null,
) {
    val density = LocalDensity.current
    val reduced = rememberReducedMotion()
    val s = scale.coerceIn(0.5f, 12f)
    val trackW = 64f * s
    val trackH = 28f * s
    val thumbW = 38f * s
    val thumbH = 24f * s
    val inset = (trackH - thumbH) / 2f
    val travel = trackW - thumbW - inset * 2f
    val target = if (checked) travel else 0f
    val animated by androidx.compose.animation.core.animateFloatAsState(
        targetValue = target,
        animationSpec = if (reduced) {
            androidx.compose.animation.core.snap<Float>()
        } else {
            androidx.compose.animation.core.spring<Float>(
                dampingRatio = androidx.compose.animation.core.Spring.DampingRatioMediumBouncy,
                stiffness = androidx.compose.animation.core.Spring.StiffnessMedium,
            )
        },
        label = "soap-toggle-thumb",
    )
    val centerX = inset + animated + thumbW / 2f
    val bleedDp = 16f
    val trackColor = Color(0xFF787880).copy(alpha = 0.22f)
    val layerPx = with(density) {
        SoapBubbleLayer(
            x = 0f,
            y = 0f,
            width = trackW.dp.toPx(),
            height = trackH.dp.toPx(),
            radius = (trackH / 2f).dp.toPx(),
            fill = if (trackW > 0f) centerX / trackW else 0f,
            track = trackColor.toCssRgb(),
            from = color.toCssRgb(),
        )
    }
    Box(
        modifier
            .size(trackW.dp, trackH.dp)
            .semantics {
                contentDescription?.let { this.contentDescription = it }
                stateDescription = if (checked) "On" else "Off"
            },
    ) {
        if (!SoapBubbleGl.ready.value) {
            Box(
                Modifier
                    .size(trackW.dp, trackH.dp)
                    .clip(CircleShape)
                    .background(if (checked) color else trackColor)
            )
            Box(
                Modifier
                    .offset(x = (inset + animated).dp, y = inset.dp)
                    .size(thumbW.dp, thumbH.dp)
                    .clip(CircleShape)
                    .background(Color.White.copy(alpha = 0.85f))
            )
        }
        SoapBubbleSurface(
            modifier = Modifier.size(trackW.dp, trackH.dp),
            radius = 9999f,
            interactive = true,
            enabled = enabled,
            onClick = { onCheckedChange(!checked) },
            role = Role.Switch,
            hostSize = DpSize(thumbW.dp, thumbH.dp),
            canvasSize = DpSize((trackW + bleedDp * 2f).dp, (trackH + bleedDp * 2f).dp),
            center = DpOffset(centerX.dp, (trackH / 2f).dp),
            layer = layerPx,
            fallback = false,
        ) {}
    }
}

@Composable
fun SoapBubbleSlider(
    value: Float,
    onValueChange: (Float) -> Unit,
    modifier: Modifier = Modifier,
    min: Float = 0f,
    max: Float = 100f,
    step: Float = 1f,
    label: String = "Value",
    from: Color = Color(0xFF0A84FF),
    scale: Float = 1f,
) {
    val density = LocalDensity.current
    val s = scale.coerceIn(0.5f, 12f)
    val thumbW = (38f * s).dp
    val thumbH = (26f * s).dp
    val trackH = (6f * s).dp
    val height = (44f * s).dp
    val bleedDp = 16f
    val fraction = if (max > min) ((value - min) / (max - min)).coerceIn(0f, 1f) else 0f
    val trackColor = Color(0xFF787880).copy(alpha = 0.2f)

    fun valueAt(x: Float, widthPx: Float): Float {
        val raw = (x / max(1f, widthPx)) * (max - min) + min
        val snapped = if (step > 0f) (Math.round(raw / step) * step) else raw
        return snapped.coerceIn(min, max)
    }

    androidx.compose.foundation.layout.BoxWithConstraints(
        modifier.fillMaxWidth().height(height)
    ) {
        val trackWidth = maxWidth
        val center = thumbW / 2f + (trackWidth - thumbW).coerceAtLeast(0.dp) * fraction
        val layerPx = with(density) {
            SoapBubbleLayer(
                x = 0f,
                y = ((height - trackH) / 2f).toPx(),
                width = trackWidth.toPx(),
                height = trackH.toPx(),
                radius = trackH.toPx() / 2f,
                fill = if (trackWidth.value > 0f) center.value / trackWidth.value else 0f,
                track = trackColor.toCssRgb(),
                from = from.toCssRgb(),
            )
        }
        if (!SoapBubbleGl.ready.value) {
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(trackH)
                    .align(Alignment.CenterStart)
                    .clip(CircleShape)
                    .background(trackColor)
            ) {
                Box(
                    Modifier
                        .fillMaxWidth(fraction)
                        .height(trackH)
                        .clip(CircleShape)
                        .background(from)
                )
            }
            Box(
                Modifier
                    .offset(x = center - thumbW / 2f)
                    .size(thumbW, thumbH)
                    .align(Alignment.CenterStart)
                    .clip(CircleShape)
                    .background(Color.White.copy(alpha = 0.85f))
            )
        }
        SoapBubbleSurface(
            modifier = Modifier
                .fillMaxWidth()
                .height(height)
                .pointerInput(min, max, step) {
                    detectTapGestures { offset ->
                        onValueChange(valueAt(offset.x, size.width.toFloat()))
                    }
                }
                .pointerInput(min, max, step) {
                    detectHorizontalDragGestures { change, _ ->
                        onValueChange(valueAt(change.position.x, size.width.toFloat()))
                    }
                }
                .semantics {
                    contentDescription = label
                    progressBarRangeInfo = androidx.compose.ui.semantics.ProgressBarRangeInfo(
                        value.coerceIn(min, max),
                        min..max,
                    )
                    setProgress { target ->
                        onValueChange(target.coerceIn(min, max))
                        true
                    }
                },
            radius = 9999f,
            interactive = true,
            hostSize = DpSize(thumbW, thumbH),
            canvasSize = DpSize(
                trackWidth + (bleedDp * 2f).dp,
                height + (bleedDp * 2f).dp,
            ),
            center = DpOffset(center, height / 2f),
            layer = layerPx,
            fallback = false,
        ) {}
    }
}

/* ------------------------------------------------------------------ */
/* Search                                                              */
/* ------------------------------------------------------------------ */

@Composable
fun SoapBubbleSearch(
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    placeholder: String = "Search",
    textStyle: TextStyle = MaterialTheme.typography.bodyMedium,
) {
    val colors = SofuuBubble.colors
    SoapBubbleSurface(
        modifier = modifier.fillMaxWidth().height(44.dp),
        radius = 9999f,
        interactive = true,
    ) {
        Row(
            Modifier.fillMaxWidth().height(44.dp).padding(start = 12.dp, end = 18.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(9.dp),
        ) {
            Icon(Icons2.Search, null, tint = colors.muted, modifier = Modifier.size(19.dp))
            Box(Modifier.weight(1f), contentAlignment = Alignment.CenterStart) {
                BasicTextField(
                    value = value,
                    onValueChange = onValueChange,
                    singleLine = true,
                    textStyle = textStyle.copy(color = colors.foreground),
                    cursorBrush = SolidColor(colors.accentDeep),
                    keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                        keyboardType = KeyboardType.Text
                    ),
                    modifier = Modifier.fillMaxWidth(),
                )
                if (value.isEmpty()) {
                    Text(placeholder, style = textStyle, color = colors.muted.copy(alpha = 0.5f))
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Menu                                                                */
/* ------------------------------------------------------------------ */

data class SoapBubbleMenuItem(
    val label: String,
    val icon: ImageVector? = null,
    val disabled: Boolean = false,
    val onSelect: () -> Unit = {},
)

/**
 * The trigger swells into a bubble panel. This keeps the web's look and
 * keyboard handling (Escape closes, arrows move) without the WebGL morph.
 */
@Composable
fun SoapBubbleMenu(
    label: String,
    items: List<SoapBubbleMenuItem>,
    modifier: Modifier = Modifier,
    triggerLabel: String? = null,
    round: Boolean = false,
) {
    var open by remember { mutableStateOf(false) }
    var active by remember { mutableStateOf(-1) }
    Box(modifier) {
        if (round) {
            SoapBubbleSurface(
                modifier = Modifier.size(44.dp),
                radius = 9999f,
                onClick = { open = !open },
                role = Role.Button,
            ) {
                Box(Modifier.fillMaxWidth().height(44.dp), contentAlignment = Alignment.Center) {
                    Text(label, style = MaterialTheme.typography.bodyMedium.copy(fontSize = 20.sp))
                }
            }
        } else {
            SoapBubbleButton(
                onClick = { open = !open },
                modifier = Modifier,
                contentColor = SofuuBubble.colors.foreground,
            ) {
                Text(label, style = MaterialTheme.typography.bodyMedium.copy(fontWeight = androidx.compose.ui.text.font.FontWeight(700)))
            }
        }
        if (open) {
            androidx.compose.ui.window.Popup(
                onDismissRequest = { open = false },
                offset = IntOffset(0, 48),
            ) {
                SoapBubbleSurface(
                    modifier = Modifier.width(220.dp),
                    radius = 24f,
                    interactive = false,
                ) {
                    androidx.compose.foundation.layout.Column(
                        Modifier.padding(vertical = 8.dp)
                    ) {
                        items.forEachIndexed { index, item ->
                            Row(
                                Modifier
                                    .fillMaxWidth()
                                    .background(
                                        if (active == index) SofuuBubble.colors.foreground.copy(alpha = 0.13f)
                                        else Color.Transparent,
                                        CircleShape,
                                    )
                                    .clickable(enabled = !item.disabled) {
                                        item.onSelect()
                                        open = false
                                    }
                                    .padding(horizontal = 20.dp, vertical = 12.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(12.dp),
                            ) {
                                if (item.icon != null) {
                                    Icon(item.icon, null, tint = SofuuBubble.colors.foreground, modifier = Modifier.size(18.dp))
                                }
                                Text(item.label, style = MaterialTheme.typography.bodyMedium, color = SofuuBubble.colors.foreground)
                            }
                        }
                    }
                }
            }
        }
    }
}
