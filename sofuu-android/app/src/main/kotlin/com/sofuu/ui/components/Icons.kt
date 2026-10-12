package com.sofuu.ui.components

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.*
import androidx.compose.material.icons.filled.*
import androidx.compose.ui.graphics.vector.ImageVector
import com.sofuu.store.Timeline

/**
 * Map the desktop's activity-icon ids onto Material symbols (the app
 * bundles material-icons-extended). Same visual vocabulary as the
 * desktop's SVG registry: read/write/edit/grep/glob/folder/terminal/web/
 * link/delegate/checklist/brain/compress/gauge/gates/rlm/warn/think/dot.
 */
fun iconFor(id: String): ImageVector = when (id) {
    Timeline.ICON_READ -> Icons.Outlined.Description
    Timeline.ICON_WRITE -> Icons.Outlined.NoteAdd
    Timeline.ICON_EDIT -> Icons.Outlined.EditNote
    Timeline.ICON_GREP -> Icons.Outlined.ManageSearch
    Timeline.ICON_GLOB -> Icons.Outlined.TravelExplore
    Timeline.ICON_FOLDER -> Icons.Outlined.FolderOpen
    Timeline.ICON_TERMINAL -> Icons.Outlined.Terminal
    Timeline.ICON_WEB -> Icons.Outlined.TravelExplore
    Timeline.ICON_LINK -> Icons.Outlined.Link
    Timeline.ICON_DELEGATE -> Icons.Outlined.Hub
    Timeline.ICON_CHECKLIST -> Icons.Outlined.Checklist
    Timeline.ICON_BRAIN -> Icons.Outlined.Psychology
    Timeline.ICON_COMPRESS -> Icons.Outlined.Compress
    Timeline.ICON_GAUGE -> Icons.Outlined.Speed
    Timeline.ICON_FRESHNESS -> Icons.Outlined.AcUnit
    Timeline.ICON_RELEVANCE -> Icons.Outlined.FilterAlt
    Timeline.ICON_SUPERVISOR -> Icons.Outlined.SupervisorAccount
    Timeline.ICON_LOOP -> Icons.Outlined.Loop
    Timeline.ICON_RLM -> Icons.Outlined.PsychologyAlt
    Timeline.ICON_WARN -> Icons.Outlined.WarningAmber
    Timeline.ICON_NOTICE -> Icons.Outlined.Info
    Timeline.ICON_PLAN -> Icons.Outlined.Map
    Timeline.ICON_THINK -> Icons.Outlined.Psychology
    else -> Icons.Filled.Circle
}

/** UI glyphs that aren't timeline rows. */
object Icons2 {
    val Plus = Icons.Outlined.Add
    val Gear = Icons.Outlined.Settings
    val Menu = Icons.Outlined.Menu
    val Send = Icons.Filled.ArrowUpward
    val Stop = Icons.Filled.Stop
    val Spark = Icons.Filled.AutoAwesome
    val Caret = Icons.Filled.ArrowDropDown
    val Clock = Icons.Outlined.Schedule
    val Close = Icons.Outlined.Close
    val Attach = Icons.Outlined.AttachFile
    val Layers = Icons.Outlined.Layers
    val Shield = Icons.Outlined.Shield
    val Folder = Icons.Outlined.Folder
    val Sidebar = Icons.Outlined.ViewSidebar
    val Chat = Icons.Outlined.ChatBubbleOutline
    val MoreVert = Icons.Outlined.MoreVert
    val ChevronRight = Icons.Filled.ChevronRight
    val ChevronDown = Icons.Filled.ExpandMore
    val Check = Icons.Filled.Check
    val Search = Icons.Outlined.Search
    val Delete = Icons.Outlined.Delete
    val Image = Icons.Outlined.Image
    val Mic = Icons.Outlined.Mic
    val Volume = Icons.Outlined.VolumeUp
    val KeyboardArrowUp = Icons.Outlined.KeyboardArrowUp
}
