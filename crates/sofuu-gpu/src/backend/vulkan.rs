#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

//! Direct Vulkan 1.1 renderer (opt-in via `SOFUU_UI_BACKEND=vulkan`).
//!
//! The loader is a ~400-line dynamic `vkGetInstanceProcAddr` shim: `libvulkan.so.1` /
//! `vulkan-1.dll` is opened at runtime, so a machine without an ICD simply falls back to the
//! GL/GLES backend. Instance arrays are std430 storage buffers indexed by `gl_InstanceIndex`;
//! clip rectangles are a second storage buffer; the glyph atlas is an `R8_UNORM` sampled image.
//! Four pipelines share the exact Metal/GLSL geometry, SDF and premultiplied-alpha math.
//!
//! Precompiled SPIR-V lives in `src/shaders/spirv/*.spv` (regenerate with
//! `src/shaders/generate_spirv.sh` from `sofuu_ui::shaders::GLSL_450_SRC`); no
//! shaderc/glslc dependency is needed at build or run time.

use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

use sofuu_ui::{ClipRect, DrawList, FrameUniforms, GlyphAtlas};

// ── Handles ────────────────────────────────────────────────────────────────────────────────
// Dispatchable Vulkan handles are pointers; non-dispatchable handles are pointer-sized values.
// `usize` is ABI-compatible with both `uint64_t` (64-bit) and the pointer form (32-bit).
type VkInstance = *mut c_void;
type VkPhysicalDevice = *mut c_void;
type VkDevice = *mut c_void;
type VkQueue = *mut c_void;
type VkCommandBuffer = *mut c_void;
type VkHandle = usize;

type VkSurfaceKHR = VkHandle;
type VkSwapchainKHR = VkHandle;
type VkImage = VkHandle;
type VkImageView = VkHandle;
type VkBuffer = VkHandle;
type VkDeviceMemory = VkHandle;
type VkPipeline = VkHandle;
type VkPipelineLayout = VkHandle;
type VkDescriptorSetLayout = VkHandle;
type VkDescriptorPool = VkHandle;
type VkDescriptorSet = VkHandle;
type VkRenderPass = VkHandle;
type VkFramebuffer = VkHandle;
type VkShaderModule = VkHandle;
type VkSampler = VkHandle;
type VkFence = VkHandle;
type VkSemaphore = VkHandle;
type VkCommandPool = VkHandle;

type VkResult = i32;
const VK_SUCCESS: VkResult = 0;
const VK_SUBOPTIMAL_KHR: VkResult = 1000001003;
const VK_ERROR_OUT_OF_DATE_KHR: VkResult = -1000001004;
const VK_WHOLE_SIZE: u64 = u64::MAX;
const VK_NULL_HANDLE: VkHandle = 0;

const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
const VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO: u32 = 9;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
const VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO: u32 = 14;
const VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO: u32 = 15;
const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
const VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO: u32 = 19;
const VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO: u32 = 20;
const VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO: u32 = 22;
const VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO: u32 = 23;
const VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO: u32 = 24;
const VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO: u32 = 26;
const VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO: u32 = 27;
const VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO: u32 = 28;
const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
const VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO: u32 = 31;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
const VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO: u32 = 37;
const VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO: u32 = 38;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
const VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO: u32 = 43;
const VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER: u32 = 45;
const VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR: u32 = 1000001000;
const VK_STRUCTURE_TYPE_PRESENT_INFO_KHR: u32 = 1000001001;
const VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR: u32 = 1000004000;
const VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR: u32 = 1000006000;
const VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR: u32 = 1000009000;
const VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR: u32 = 1000014000;

const VK_QUEUE_GRAPHICS_BIT: u32 = 1;
const VK_FORMAT_B8G8R8A8_UNORM: u32 = 44;
const VK_FORMAT_R8G8B8A8_UNORM: u32 = 37;
const VK_FORMAT_R8_UNORM: u32 = 9;
const VK_COLOR_SPACE_SRGB_NONLINEAR_KHR: u32 = 0;
const VK_PRESENT_MODE_FIFO_KHR: u32 = 2;
const VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR: u32 = 1;
const VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR: u32 = 0x4;
const VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR: u32 = 2;
const VK_IMAGE_USAGE_TRANSFER_DST_BIT: u32 = 0x2;
const VK_IMAGE_USAGE_SAMPLED_BIT: u32 = 0x4;
const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x1;
const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x20;
const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;
const VK_IMAGE_TILING_OPTIMAL: u32 = 0;
const VK_IMAGE_TYPE_2D: u32 = 0;
const VK_IMAGE_VIEW_TYPE_2D: u32 = 1;
const VK_SAMPLE_COUNT_1_BIT: u32 = 1;
const VK_IMAGE_LAYOUT_UNDEFINED: u32 = 0;
const VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: u32 = 5;
const VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: u32 = 6;
const VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: u32 = 1000001002;
const VK_IMAGE_ASPECT_COLOR_BIT: u32 = 1;
const VK_ACCESS_TRANSFER_WRITE_BIT: u32 = 0x1000;
const VK_ACCESS_SHADER_READ_BIT: u32 = 0x20;
const VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT: u32 = 0x400;
const VK_PIPELINE_STAGE_TRANSFER_BIT: u32 = 0x1000;
const VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT: u32 = 0x400;
const VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT: u32 = 0x80;
const VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT: u32 = 0x1;
const VK_SUBPASS_EXTERNAL: u32 = 0xFFFFFFFF;
const VK_ATTACHMENT_LOAD_OP_CLEAR: u32 = 1;
const VK_ATTACHMENT_STORE_OP_STORE: u32 = 0;
const VK_ATTACHMENT_STORE_OP_DONT_CARE: u32 = 1;
const VK_PIPELINE_BIND_POINT_GRAPHICS: u32 = 0;
const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
const VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: u32 = 1;
const VK_SHADER_STAGE_VERTEX_BIT: u32 = 0x1;
const VK_SHADER_STAGE_FRAGMENT_BIT: u32 = 0x10;
const VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: u32 = 4;
const VK_POLYGON_MODE_FILL: u32 = 0;
const VK_CULL_MODE_NONE: u32 = 0;
const VK_FRONT_FACE_COUNTER_CLOCKWISE: u32 = 0;
const VK_BLEND_FACTOR_ONE: u32 = 1;
const VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: u32 = 6;
const VK_BLEND_OP_ADD: u32 = 0;
const VK_COLOR_COMPONENT_RGBA_BIT: u32 = 0xF;
const VK_DYNAMIC_STATE_VIEWPORT: u32 = 1;
const VK_DYNAMIC_STATE_SCISSOR: u32 = 2;
const VK_FILTER_LINEAR: u32 = 1;
const VK_SAMPLER_MIPMAP_MODE_NEAREST: u32 = 0;
const VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: u32 = 2;
const VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK: u32 = 1;
const VK_COMPARE_OP_NEVER: u32 = 0;
const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 1;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 2;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 4;
const VK_FENCE_CREATE_SIGNALED_BIT: u32 = 1;
const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 2;
const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 1;

const FRAMES_IN_FLIGHT: usize = 2;

const VK_KHR_SURFACE_EXTENSION_NAME: &CStr = c"VK_KHR_surface";
const VK_KHR_SWAPCHAIN_EXTENSION_NAME: &CStr = c"VK_KHR_swapchain";
const VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME: &CStr = c"VK_KHR_wayland_surface";
const VK_KHR_XLIB_SURFACE_EXTENSION_NAME: &CStr = c"VK_KHR_xlib_surface";
const VK_KHR_WIN32_SURFACE_EXTENSION_NAME: &CStr = c"VK_KHR_win32_surface";
const VK_KHR_ANDROID_SURFACE_EXTENSION_NAME: &CStr = c"VK_KHR_android_surface";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceKind {
    Wayland,
    Xlib,
    Win32,
    Android,
}

#[derive(Clone, Copy)]
pub struct NativeSurface {
    pub kind: SurfaceKind,
    pub display: *mut c_void,
    pub window: *mut c_void,
}

// ── C ABI structs ──────────────────────────────────────────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
struct VkExtent2D {
    width: u32,
    height: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkExtent3D {
    width: u32,
    height: u32,
    depth: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkOffset2D {
    x: i32,
    y: i32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkOffset3D {
    x: i32,
    y: i32,
    z: i32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkRect2D {
    offset: VkOffset2D,
    extent: VkExtent2D,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkViewport {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    minDepth: f32,
    maxDepth: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkApplicationInfo {
    sType: u32,
    pNext: *const c_void,
    pApplicationName: *const c_char,
    applicationVersion: u32,
    pEngineName: *const c_char,
    engineVersion: u32,
    apiVersion: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkInstanceCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    pApplicationInfo: *const VkApplicationInfo,
    enabledLayerCount: u32,
    ppEnabledLayerNames: *const *const c_char,
    enabledExtensionCount: u32,
    ppEnabledExtensionNames: *const *const c_char,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkExtensionProperties {
    extensionName: [c_char; 256],
    specVersion: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkQueueFamilyProperties {
    queueFlags: u32,
    queueCount: u32,
    timestampValidBits: u32,
    minImageTransferGranularity: VkExtent3D,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkDeviceQueueCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    queueFamilyIndex: u32,
    queueCount: u32,
    pQueuePriorities: *const f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkDeviceCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    queueCreateInfoCount: u32,
    pQueueCreateInfos: *const VkDeviceQueueCreateInfo,
    enabledLayerCount: u32,
    ppEnabledLayerNames: *const *const c_char,
    enabledExtensionCount: u32,
    ppEnabledExtensionNames: *const *const c_char,
    pEnabledFeatures: *const c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkSurfaceCapabilitiesKHR {
    minImageCount: u32,
    maxImageCount: u32,
    currentExtent: VkExtent2D,
    minImageExtent: VkExtent2D,
    maxImageExtent: VkExtent2D,
    maxImageArrayLayers: u32,
    supportedTransforms: u32,
    currentTransform: u32,
    supportedCompositeAlpha: u32,
    supportedUsageFlags: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkSurfaceFormatKHR {
    format: u32,
    colorSpace: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkSwapchainCreateInfoKHR {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    surface: VkSurfaceKHR,
    minImageCount: u32,
    imageFormat: u32,
    imageColorSpace: u32,
    imageExtent: VkExtent2D,
    imageArrayLayers: u32,
    imageUsage: u32,
    imageSharingMode: u32,
    queueFamilyIndexCount: u32,
    pQueueFamilyIndices: *const u32,
    preTransform: u32,
    compositeAlpha: u32,
    presentMode: u32,
    clipped: u32,
    oldSwapchain: VkSwapchainKHR,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkImageCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    imageType: u32,
    format: u32,
    extent: VkExtent3D,
    mipLevels: u32,
    arrayLayers: u32,
    samples: u32,
    tiling: u32,
    usage: u32,
    sharingMode: u32,
    queueFamilyIndexCount: u32,
    pQueueFamilyIndices: *const u32,
    initialLayout: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkImageViewCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    image: VkImage,
    viewType: u32,
    format: u32,
    components: [u32; 4],
    subresourceRange: VkImageSubresourceRange,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkImageSubresourceRange {
    aspectMask: u32,
    baseMipLevel: u32,
    levelCount: u32,
    baseArrayLayer: u32,
    layerCount: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkBufferCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    size: u64,
    usage: u32,
    sharingMode: u32,
    queueFamilyIndexCount: u32,
    pQueueFamilyIndices: *const u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryRequirements {
    size: u64,
    alignment: u64,
    memoryTypeBits: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryAllocateInfo {
    sType: u32,
    pNext: *const c_void,
    allocationSize: u64,
    memoryTypeIndex: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkPhysicalDeviceMemoryProperties {
    memoryTypeCount: u32,
    memoryTypes: [VkMemoryType; 32],
    memoryHeapCount: u32,
    memoryHeaps: [VkMemoryHeap; 16],
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryType {
    propertyFlags: u32,
    heapIndex: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryHeap {
    size: u64,
    flags: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkBufferImageCopy {
    bufferOffset: u64,
    bufferRowLength: u32,
    bufferImageHeight: u32,
    imageSubresource: VkImageSubresourceLayers,
    imageOffset: VkOffset3D,
    imageExtent: VkExtent3D,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkImageSubresourceLayers {
    aspectMask: u32,
    mipLevel: u32,
    baseArrayLayer: u32,
    layerCount: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkImageMemoryBarrier {
    sType: u32,
    pNext: *const c_void,
    srcAccessMask: u32,
    dstAccessMask: u32,
    oldLayout: u32,
    newLayout: u32,
    srcQueueFamilyIndex: u32,
    dstQueueFamilyIndex: u32,
    image: VkImage,
    subresourceRange: VkImageSubresourceRange,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkShaderModuleCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    codeSize: usize,
    pCode: *const u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkPipelineShaderStageCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    stage: u32,
    module: VkShaderModule,
    pName: *const c_char,
    pSpecializationInfo: *const c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkPipelineVertexInputStateCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    vertexBindingDescriptionCount: u32,
    pVertexBindingDescriptions: *const c_void,
    vertexAttributeDescriptionCount: u32,
    pVertexAttributeDescriptions: *const c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkPipelineInputAssemblyStateCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    topology: u32,
    primitiveRestartEnable: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkPipelineViewportStateCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    viewportCount: u32,
    pViewports: *const VkViewport,
    scissorCount: u32,
    pScissors: *const VkRect2D,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkPipelineRasterizationStateCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    depthClampEnable: u32,
    rasterizerDiscardEnable: u32,
    polygonMode: u32,
    cullMode: u32,
    frontFace: u32,
    depthBiasEnable: u32,
    depthBiasConstantFactor: f32,
    depthBiasClamp: f32,
    depthBiasSlopeFactor: f32,
    lineWidth: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkPipelineMultisampleStateCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    rasterizationSamples: u32,
    sampleShadingEnable: u32,
    minSampleShading: f32,
    pSampleMask: *const u32,
    alphaToCoverageEnable: u32,
    alphaToOneEnable: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkPipelineColorBlendAttachmentState {
    blendEnable: u32,
    srcColorBlendFactor: u32,
    dstColorBlendFactor: u32,
    colorBlendOp: u32,
    srcAlphaBlendFactor: u32,
    dstAlphaBlendFactor: u32,
    alphaBlendOp: u32,
    colorWriteMask: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkPipelineColorBlendStateCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    logicOpEnable: u32,
    logicOp: u32,
    attachmentCount: u32,
    pAttachments: *const VkPipelineColorBlendAttachmentState,
    blendConstants: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkPipelineDynamicStateCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    dynamicStateCount: u32,
    pDynamicStates: *const u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkPushConstantRange {
    stageFlags: u32,
    offset: u32,
    size: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkPipelineLayoutCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    setLayoutCount: u32,
    pSetLayouts: *const VkDescriptorSetLayout,
    pushConstantRangeCount: u32,
    pPushConstantRanges: *const VkPushConstantRange,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkGraphicsPipelineCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    stageCount: u32,
    pStages: *const VkPipelineShaderStageCreateInfo,
    pVertexInputState: *const VkPipelineVertexInputStateCreateInfo,
    pInputAssemblyState: *const VkPipelineInputAssemblyStateCreateInfo,
    pTessellationState: *const c_void,
    pViewportState: *const VkPipelineViewportStateCreateInfo,
    pRasterizationState: *const VkPipelineRasterizationStateCreateInfo,
    pMultisampleState: *const VkPipelineMultisampleStateCreateInfo,
    pDepthStencilState: *const c_void,
    pColorBlendState: *const VkPipelineColorBlendStateCreateInfo,
    pDynamicState: *const VkPipelineDynamicStateCreateInfo,
    layout: VkPipelineLayout,
    renderPass: VkRenderPass,
    subpass: u32,
    basePipelineHandle: VkPipeline,
    basePipelineIndex: i32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkAttachmentDescription {
    flags: u32,
    format: u32,
    samples: u32,
    loadOp: u32,
    storeOp: u32,
    stencilLoadOp: u32,
    stencilStoreOp: u32,
    initialLayout: u32,
    finalLayout: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkAttachmentReference {
    attachment: u32,
    layout: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkSubpassDescription {
    flags: u32,
    pipelineBindPoint: u32,
    inputAttachmentCount: u32,
    pInputAttachments: *const VkAttachmentReference,
    colorAttachmentCount: u32,
    pColorAttachments: *const VkAttachmentReference,
    pResolveAttachments: *const VkAttachmentReference,
    pDepthStencilAttachment: *const VkAttachmentReference,
    preserveAttachmentCount: u32,
    pPreserveAttachments: *const u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkSubpassDependency {
    srcSubpass: u32,
    dstSubpass: u32,
    srcStageMask: u32,
    dstStageMask: u32,
    srcAccessMask: u32,
    dstAccessMask: u32,
    dependencyFlags: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkRenderPassCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    attachmentCount: u32,
    pAttachments: *const VkAttachmentDescription,
    subpassCount: u32,
    pSubpasses: *const VkSubpassDescription,
    dependencyCount: u32,
    pDependencies: *const VkSubpassDependency,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkFramebufferCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    renderPass: VkRenderPass,
    attachmentCount: u32,
    pAttachments: *const VkImageView,
    width: u32,
    height: u32,
    layers: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkSamplerCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    magFilter: u32,
    minFilter: u32,
    mipmapMode: u32,
    addressModeU: u32,
    addressModeV: u32,
    addressModeW: u32,
    mipLodBias: f32,
    anisotropyEnable: u32,
    maxAnisotropy: f32,
    compareEnable: u32,
    compareOp: u32,
    minLod: f32,
    maxLod: f32,
    borderColor: u32,
    unnormalizedCoordinates: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkDescriptorSetLayoutBinding {
    binding: u32,
    descriptorType: u32,
    descriptorCount: u32,
    stageFlags: u32,
    pImmutableSamplers: *const c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkDescriptorSetLayoutCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    bindingCount: u32,
    pBindings: *const VkDescriptorSetLayoutBinding,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkDescriptorPoolSize {
    ty: u32,
    descriptorCount: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkDescriptorPoolCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    maxSets: u32,
    poolSizeCount: u32,
    pPoolSizes: *const VkDescriptorPoolSize,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkDescriptorSetAllocateInfo {
    sType: u32,
    pNext: *const c_void,
    descriptorPool: VkDescriptorPool,
    descriptorSetCount: u32,
    pSetLayouts: *const VkDescriptorSetLayout,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkDescriptorBufferInfo {
    buffer: VkBuffer,
    offset: u64,
    range: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkDescriptorImageInfo {
    sampler: VkSampler,
    imageView: VkImageView,
    imageLayout: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkWriteDescriptorSet {
    sType: u32,
    pNext: *const c_void,
    dstSet: VkDescriptorSet,
    dstBinding: u32,
    dstArrayElement: u32,
    descriptorCount: u32,
    descriptorType: u32,
    pImageInfo: *const VkDescriptorImageInfo,
    pBufferInfo: *const VkDescriptorBufferInfo,
    pTexelBufferView: *const c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkCommandPoolCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    queueFamilyIndex: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkCommandBufferAllocateInfo {
    sType: u32,
    pNext: *const c_void,
    commandPool: VkCommandPool,
    level: u32,
    commandBufferCount: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkCommandBufferBeginInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    pInheritanceInfo: *const c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkClearValue {
    color: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkRenderPassBeginInfo {
    sType: u32,
    pNext: *const c_void,
    renderPass: VkRenderPass,
    framebuffer: VkFramebuffer,
    renderArea: VkRect2D,
    clearValueCount: u32,
    pClearValues: *const VkClearValue,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkFenceCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkSemaphoreCreateInfo {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkSubmitInfo {
    sType: u32,
    pNext: *const c_void,
    waitSemaphoreCount: u32,
    pWaitSemaphores: *const VkSemaphore,
    pWaitDstStageMask: *const u32,
    commandBufferCount: u32,
    pCommandBuffers: *const VkCommandBuffer,
    signalSemaphoreCount: u32,
    pSignalSemaphores: *const VkSemaphore,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkPresentInfoKHR {
    sType: u32,
    pNext: *const c_void,
    waitSemaphoreCount: u32,
    pWaitSemaphores: *const VkSemaphore,
    swapchainCount: u32,
    pSwapchains: *const VkSwapchainKHR,
    pImageIndices: *const u32,
    pResults: *mut VkResult,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkWaylandSurfaceCreateInfoKHR {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    display: *mut c_void,
    surface: *mut c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkXlibSurfaceCreateInfoKHR {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    dpy: *mut c_void,
    window: usize,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkWin32SurfaceCreateInfoKHR {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    hinstance: *mut c_void,
    hwnd: *mut c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct VkAndroidSurfaceCreateInfoKHR {
    sType: u32,
    pNext: *const c_void,
    flags: u32,
    window: *mut c_void,
}

type PFN_vkGetInstanceProcAddr =
    unsafe extern "system" fn(VkInstance, *const c_char) -> *const c_void;
type PFN_vkGetDeviceProcAddr = unsafe extern "system" fn(VkDevice, *const c_char) -> *const c_void;
type PFN_vkCreateWaylandSurfaceKHR = unsafe extern "system" fn(
    VkInstance,
    *const VkWaylandSurfaceCreateInfoKHR,
    *const c_void,
    *mut VkSurfaceKHR,
) -> VkResult;
type PFN_vkCreateXlibSurfaceKHR = unsafe extern "system" fn(
    VkInstance,
    *const VkXlibSurfaceCreateInfoKHR,
    *const c_void,
    *mut VkSurfaceKHR,
) -> VkResult;
type PFN_vkCreateWin32SurfaceKHR = unsafe extern "system" fn(
    VkInstance,
    *const VkWin32SurfaceCreateInfoKHR,
    *const c_void,
    *mut VkSurfaceKHR,
) -> VkResult;
type PFN_vkCreateAndroidSurfaceKHR = unsafe extern "system" fn(
    VkInstance,
    *const VkAndroidSurfaceCreateInfoKHR,
    *const c_void,
    *mut VkSurfaceKHR,
) -> VkResult;

macro_rules! vk_struct {
    ($name:ident { $( fn $fn:ident($($arg:ident: $ty:ty),*) -> $ret:ty; )* }) => {
        #[derive(Clone, Copy)]
        #[allow(dead_code)]
        struct $name {
            $( $fn: unsafe extern "system" fn($($ty),*) -> $ret, )*
        }

        impl $name {
            /// Resolve every entry point through `get`, failing with the precise missing name.
            unsafe fn load(get: &dyn Fn(&CStr) -> *const c_void) -> Result<Self, String> {
                $(
                    let $fn: unsafe extern "system" fn($($ty),*) -> $ret = {
                        let c = CString::new(stringify!($fn)).unwrap();
                        let p = get(&c);
                        if p.is_null() {
                            return Err(format!(
                                "E_UI_VULKAN_MISSING: required entry point `{}` is unavailable",
                                stringify!($fn)
                            ));
                        }
                        std::mem::transmute::<*const c_void, unsafe extern "system" fn($($ty),*) -> $ret>(p)
                    };
                )*
                Ok(Self { $($fn),* })
            }
        }
    };
}

vk_struct!(VkGlobalFns {
    fn vkCreateInstance(info: *const VkInstanceCreateInfo, allocator: *const c_void, out: *mut VkInstance) -> VkResult;
    fn vkEnumerateInstanceExtensionProperties(layer: *const c_char, count: *mut u32, props: *mut VkExtensionProperties) -> VkResult;
});

vk_struct!(VkInstanceFns {
    fn vkDestroyInstance(instance: VkInstance, allocator: *const c_void) -> ();
    fn vkGetDeviceProcAddr(device: VkDevice, name: *const c_char) -> *const c_void;
    fn vkEnumeratePhysicalDevices(instance: VkInstance, count: *mut u32, devices: *mut VkPhysicalDevice) -> VkResult;
    fn vkGetPhysicalDeviceQueueFamilyProperties(device: VkPhysicalDevice, count: *mut u32, props: *mut VkQueueFamilyProperties) -> ();
    fn vkGetPhysicalDeviceMemoryProperties(device: VkPhysicalDevice, props: *mut VkPhysicalDeviceMemoryProperties) -> ();
    fn vkGetPhysicalDeviceSurfaceSupportKHR(device: VkPhysicalDevice, family: u32, surface: VkSurfaceKHR, supported: *mut u32) -> VkResult;
    fn vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device: VkPhysicalDevice, surface: VkSurfaceKHR, caps: *mut VkSurfaceCapabilitiesKHR) -> VkResult;
    fn vkGetPhysicalDeviceSurfaceFormatsKHR(device: VkPhysicalDevice, surface: VkSurfaceKHR, count: *mut u32, formats: *mut VkSurfaceFormatKHR) -> VkResult;
    fn vkGetPhysicalDeviceSurfacePresentModesKHR(device: VkPhysicalDevice, surface: VkSurfaceKHR, count: *mut u32, modes: *mut u32) -> VkResult;
    fn vkCreateDevice(device: VkPhysicalDevice, info: *const VkDeviceCreateInfo, allocator: *const c_void, out: *mut VkDevice) -> VkResult;
    fn vkDestroySurfaceKHR(instance: VkInstance, surface: VkSurfaceKHR, allocator: *const c_void) -> ();
});

vk_struct!(VkDeviceFns {
    fn vkDestroyDevice(device: VkDevice, allocator: *const c_void) -> ();
    fn vkGetDeviceQueue(device: VkDevice, family: u32, index: u32, out: *mut VkQueue) -> ();
    fn vkDeviceWaitIdle(device: VkDevice) -> VkResult;
    fn vkCreateSwapchainKHR(device: VkDevice, info: *const VkSwapchainCreateInfoKHR, allocator: *const c_void, out: *mut VkSwapchainKHR) -> VkResult;
    fn vkDestroySwapchainKHR(device: VkDevice, swapchain: VkSwapchainKHR, allocator: *const c_void) -> ();
    fn vkGetSwapchainImagesKHR(device: VkDevice, swapchain: VkSwapchainKHR, count: *mut u32, images: *mut VkImage) -> VkResult;
    fn vkCreateImageView(device: VkDevice, info: *const VkImageViewCreateInfo, allocator: *const c_void, out: *mut VkImageView) -> VkResult;
    fn vkDestroyImageView(device: VkDevice, view: VkImageView, allocator: *const c_void) -> ();
    fn vkCreateRenderPass(device: VkDevice, info: *const VkRenderPassCreateInfo, allocator: *const c_void, out: *mut VkRenderPass) -> VkResult;
    fn vkDestroyRenderPass(device: VkDevice, pass: VkRenderPass, allocator: *const c_void) -> ();
    fn vkCreateFramebuffer(device: VkDevice, info: *const VkFramebufferCreateInfo, allocator: *const c_void, out: *mut VkFramebuffer) -> VkResult;
    fn vkDestroyFramebuffer(device: VkDevice, fb: VkFramebuffer, allocator: *const c_void) -> ();
    fn vkCreateShaderModule(device: VkDevice, info: *const VkShaderModuleCreateInfo, allocator: *const c_void, out: *mut VkShaderModule) -> VkResult;
    fn vkDestroyShaderModule(device: VkDevice, module: VkShaderModule, allocator: *const c_void) -> ();
    fn vkCreatePipelineLayout(device: VkDevice, info: *const VkPipelineLayoutCreateInfo, allocator: *const c_void, out: *mut VkPipelineLayout) -> VkResult;
    fn vkDestroyPipelineLayout(device: VkDevice, layout: VkPipelineLayout, allocator: *const c_void) -> ();
    fn vkCreateGraphicsPipelines(device: VkDevice, cache: VkHandle, count: u32, infos: *const VkGraphicsPipelineCreateInfo, allocator: *const c_void, out: *mut VkPipeline) -> VkResult;
    fn vkDestroyPipeline(device: VkDevice, pipeline: VkPipeline, allocator: *const c_void) -> ();
    fn vkCreateBuffer(device: VkDevice, info: *const VkBufferCreateInfo, allocator: *const c_void, out: *mut VkBuffer) -> VkResult;
    fn vkDestroyBuffer(device: VkDevice, buffer: VkBuffer, allocator: *const c_void) -> ();
    fn vkGetBufferMemoryRequirements(device: VkDevice, buffer: VkBuffer, out: *mut VkMemoryRequirements) -> ();
    fn vkAllocateMemory(device: VkDevice, info: *const VkMemoryAllocateInfo, allocator: *const c_void, out: *mut VkDeviceMemory) -> VkResult;
    fn vkFreeMemory(device: VkDevice, memory: VkDeviceMemory, allocator: *const c_void) -> ();
    fn vkBindBufferMemory(device: VkDevice, buffer: VkBuffer, memory: VkDeviceMemory, offset: u64) -> VkResult;
    fn vkMapMemory(device: VkDevice, memory: VkDeviceMemory, offset: u64, size: u64, flags: u32, out: *mut *mut c_void) -> VkResult;
    fn vkUnmapMemory(device: VkDevice, memory: VkDeviceMemory) -> ();
    fn vkCreateImage(device: VkDevice, info: *const VkImageCreateInfo, allocator: *const c_void, out: *mut VkImage) -> VkResult;
    fn vkDestroyImage(device: VkDevice, image: VkImage, allocator: *const c_void) -> ();
    fn vkGetImageMemoryRequirements(device: VkDevice, image: VkImage, out: *mut VkMemoryRequirements) -> ();
    fn vkBindImageMemory(device: VkDevice, image: VkImage, memory: VkDeviceMemory, offset: u64) -> VkResult;
    fn vkCreateSampler(device: VkDevice, info: *const VkSamplerCreateInfo, allocator: *const c_void, out: *mut VkSampler) -> VkResult;
    fn vkDestroySampler(device: VkDevice, sampler: VkSampler, allocator: *const c_void) -> ();
    fn vkCreateDescriptorSetLayout(device: VkDevice, info: *const VkDescriptorSetLayoutCreateInfo, allocator: *const c_void, out: *mut VkDescriptorSetLayout) -> VkResult;
    fn vkDestroyDescriptorSetLayout(device: VkDevice, layout: VkDescriptorSetLayout, allocator: *const c_void) -> ();
    fn vkCreateDescriptorPool(device: VkDevice, info: *const VkDescriptorPoolCreateInfo, allocator: *const c_void, out: *mut VkDescriptorPool) -> VkResult;
    fn vkDestroyDescriptorPool(device: VkDevice, pool: VkDescriptorPool, allocator: *const c_void) -> ();
    fn vkAllocateDescriptorSets(device: VkDevice, info: *const VkDescriptorSetAllocateInfo, out: *mut VkDescriptorSet) -> VkResult;
    fn vkUpdateDescriptorSets(device: VkDevice, count: u32, writes: *const VkWriteDescriptorSet, copyCount: u32, copies: *const c_void) -> ();
    fn vkCreateCommandPool(device: VkDevice, info: *const VkCommandPoolCreateInfo, allocator: *const c_void, out: *mut VkCommandPool) -> VkResult;
    fn vkDestroyCommandPool(device: VkDevice, pool: VkCommandPool, allocator: *const c_void) -> ();
    fn vkAllocateCommandBuffers(device: VkDevice, info: *const VkCommandBufferAllocateInfo, out: *mut VkCommandBuffer) -> VkResult;
    fn vkBeginCommandBuffer(cmd: VkCommandBuffer, info: *const VkCommandBufferBeginInfo) -> VkResult;
    fn vkEndCommandBuffer(cmd: VkCommandBuffer) -> VkResult;
    fn vkResetCommandBuffer(cmd: VkCommandBuffer, flags: u32) -> VkResult;
    fn vkCmdBeginRenderPass(cmd: VkCommandBuffer, info: *const VkRenderPassBeginInfo, contents: u32) -> ();
    fn vkCmdEndRenderPass(cmd: VkCommandBuffer) -> ();
    fn vkCmdBindPipeline(cmd: VkCommandBuffer, bindPoint: u32, pipeline: VkPipeline) -> ();
    fn vkCmdBindDescriptorSets(cmd: VkCommandBuffer, bindPoint: u32, layout: VkPipelineLayout, firstSet: u32, setCount: u32, sets: *const VkDescriptorSet, dynamicOffsetCount: u32, dynamicOffsets: *const u32) -> ();
    fn vkCmdPushConstants(cmd: VkCommandBuffer, layout: VkPipelineLayout, stageFlags: u32, offset: u32, size: u32, values: *const c_void) -> ();
    fn vkCmdSetViewport(cmd: VkCommandBuffer, first: u32, count: u32, viewports: *const VkViewport) -> ();
    fn vkCmdSetScissor(cmd: VkCommandBuffer, first: u32, count: u32, scissors: *const VkRect2D) -> ();
    fn vkCmdDraw(cmd: VkCommandBuffer, vertexCount: u32, instanceCount: u32, firstVertex: u32, firstInstance: u32) -> ();
    fn vkCmdCopyBufferToImage(cmd: VkCommandBuffer, buffer: VkBuffer, image: VkImage, layout: u32, count: u32, regions: *const VkBufferImageCopy) -> ();
    fn vkCmdPipelineBarrier(cmd: VkCommandBuffer, srcStage: u32, dstStage: u32, flags: u32, memoryBarrierCount: u32, memoryBarriers: *const c_void, bufferBarrierCount: u32, bufferBarriers: *const c_void, imageBarrierCount: u32, imageBarriers: *const VkImageMemoryBarrier) -> ();
    fn vkQueueSubmit(queue: VkQueue, count: u32, submits: *const VkSubmitInfo, fence: VkFence) -> VkResult;
    fn vkQueueWaitIdle(queue: VkQueue) -> VkResult;
    fn vkCreateSemaphore(device: VkDevice, info: *const VkSemaphoreCreateInfo, allocator: *const c_void, out: *mut VkSemaphore) -> VkResult;
    fn vkDestroySemaphore(device: VkDevice, semaphore: VkSemaphore, allocator: *const c_void) -> ();
    fn vkCreateFence(device: VkDevice, info: *const VkFenceCreateInfo, allocator: *const c_void, out: *mut VkFence) -> VkResult;
    fn vkDestroyFence(device: VkDevice, fence: VkFence, allocator: *const c_void) -> ();
    fn vkWaitForFences(device: VkDevice, count: u32, fences: *const VkFence, waitAll: u32, timeout: u64) -> VkResult;
    fn vkResetFences(device: VkDevice, count: u32, fences: *const VkFence) -> VkResult;
    fn vkAcquireNextImageKHR(device: VkDevice, swapchain: VkSwapchainKHR, timeout: u64, semaphore: VkSemaphore, fence: VkFence, index: *mut u32) -> VkResult;
    fn vkQueuePresentKHR(queue: VkQueue, info: *const VkPresentInfoKHR) -> VkResult;
});

struct DynLib {
    handle: *mut c_void,
}

impl DynLib {
    unsafe fn open(names: &[&str]) -> Option<Self> {
        for name in names {
            let c = CString::new(*name).ok()?;
            #[cfg(unix)]
            let handle = libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
            #[cfg(windows)]
            let handle = LoadLibraryA(c.as_ptr());
            if !handle.is_null() {
                return Some(Self { handle });
            }
        }
        None
    }
    unsafe fn sym(&self, name: &str) -> *mut c_void {
        let c = CString::new(name).ok().unwrap();
        #[cfg(unix)]
        let p = libc::dlsym(self.handle, c.as_ptr());
        #[cfg(windows)]
        let p = GetProcAddress(self.handle, c.as_ptr());
        p
    }
}

#[cfg(windows)]
extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}

// ── Small helpers ──────────────────────────────────────────────────────────────────────────
fn vk_check(result: VkResult, what: &str) -> Result<(), String> {
    if result == VK_SUCCESS {
        Ok(())
    } else {
        Err(format!("E_UI_VULKAN: {what} failed (VkResult {result})"))
    }
}

unsafe fn find_memory_type(
    props: &VkPhysicalDeviceMemoryProperties,
    bits: u32,
    required: u32,
) -> Option<u32> {
    for i in 0..props.memoryTypeCount {
        let ty = props.memoryTypes[i as usize];
        if bits & (1u32 << i) != 0 && ty.propertyFlags & required == required {
            return Some(i);
        }
    }
    None
}

fn bytes_of<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

const SHADER_QUAD_VERT: &[u8] = include_bytes!("../shaders/spirv/quad.vert.spv");
const SHADER_QUAD_FRAG: &[u8] = include_bytes!("../shaders/spirv/quad.frag.spv");
const SHADER_RIBBON_VERT: &[u8] = include_bytes!("../shaders/spirv/ribbon.vert.spv");
const SHADER_RIBBON_FRAG: &[u8] = include_bytes!("../shaders/spirv/ribbon.frag.spv");
const SHADER_STROKE_VERT: &[u8] = include_bytes!("../shaders/spirv/stroke.vert.spv");
const SHADER_STROKE_FRAG: &[u8] = include_bytes!("../shaders/spirv/stroke.frag.spv");
const SHADER_GLYPH_VERT: &[u8] = include_bytes!("../shaders/spirv/glyph.vert.spv");
const SHADER_GLYPH_FRAG: &[u8] = include_bytes!("../shaders/spirv/glyph.frag.spv");
const SHADER_EFFECT_VERT: &[u8] = include_bytes!("../shaders/spirv/effect.vert.spv");
const SHADER_EFFECT_FRAG: &[u8] = include_bytes!("../shaders/spirv/effect.frag.spv");
const SHADER_PARTICLE_VERT: &[u8] = include_bytes!("../shaders/spirv/particle.vert.spv");
const SHADER_PARTICLE_FRAG: &[u8] = include_bytes!("../shaders/spirv/particle.frag.spv");

#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    viewport: [f32; 2],
    atlas: [f32; 2],
    scale: f32,
    time: f32,
    clip_count: u32,
    /// First instance for this draw. The shader indexes `instances[base + gl_InstanceIndex]`
    /// instead of relying on `firstInstance`/`shaderDrawParameters`.
    base_instance: u32,
}

#[derive(Clone, Copy)]
struct HostBuffer {
    handle: VkBuffer,
    memory: VkDeviceMemory,
    mapped: *mut u8,
    capacity: usize,
}

struct Atlas {
    image: VkImage,
    memory: VkDeviceMemory,
    view: VkImageView,
    sampler: VkSampler,
    width: u32,
    height: u32,
}

struct Swapchain {
    handle: VkSwapchainKHR,
    _images: Vec<VkImage>,
    views: Vec<VkImageView>,
    framebuffers: Vec<VkFramebuffer>,
    extent: VkExtent2D,
    format: u32,
}

struct FrameResources {
    quads: HostBuffer,
    ribbons: HostBuffer,
    strokes: HostBuffer,
    glyphs: HostBuffer,
    effects: HostBuffer,
    particles: HostBuffer,
    clips: HostBuffer,
}

impl FrameResources {
    const fn empty() -> Self {
        Self {
            quads: HostBuffer::empty(),
            ribbons: HostBuffer::empty(),
            strokes: HostBuffer::empty(),
            glyphs: HostBuffer::empty(),
            effects: HostBuffer::empty(),
            particles: HostBuffer::empty(),
            clips: HostBuffer::empty(),
        }
    }
}

impl HostBuffer {
    const fn empty() -> Self {
        Self {
            handle: VK_NULL_HANDLE,
            memory: VK_NULL_HANDLE,
            mapped: ptr::null_mut(),
            capacity: 0,
        }
    }
}

fn create_host_buffer(
    f: &VkDeviceFns,
    dev: VkDevice,
    mem: &VkPhysicalDeviceMemoryProperties,
    usage: u32,
    capacity: usize,
) -> Result<HostBuffer, String> {
    unsafe {
        let info = VkBufferCreateInfo {
            sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            size: capacity as u64,
            usage,
            sharingMode: VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
        };
        let mut handle: VkBuffer = VK_NULL_HANDLE;
        vk_check(
            (f.vkCreateBuffer)(dev, &info, ptr::null(), &mut handle),
            "vkCreateBuffer",
        )?;
        let mut req: VkMemoryRequirements = std::mem::zeroed();
        (f.vkGetBufferMemoryRequirements)(dev, handle, &mut req);
        let ty = find_memory_type(
            mem,
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        .ok_or_else(|| "E_UI_VULKAN: no host-visible coherent memory type".to_string())?;
        let alloc = VkMemoryAllocateInfo {
            sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: ptr::null(),
            allocationSize: req.size,
            memoryTypeIndex: ty,
        };
        let mut memory: VkDeviceMemory = VK_NULL_HANDLE;
        vk_check(
            (f.vkAllocateMemory)(dev, &alloc, ptr::null(), &mut memory),
            "vkAllocateMemory",
        )?;
        vk_check(
            (f.vkBindBufferMemory)(dev, handle, memory, 0),
            "vkBindBufferMemory",
        )?;
        let mut mapped: *mut c_void = ptr::null_mut();
        vk_check(
            (f.vkMapMemory)(dev, memory, 0, VK_WHOLE_SIZE, 0, &mut mapped),
            "vkMapMemory",
        )?;
        Ok(HostBuffer {
            handle,
            memory,
            mapped: mapped.cast(),
            capacity,
        })
    }
}

fn destroy_host_buffer(f: &VkDeviceFns, dev: VkDevice, b: &mut HostBuffer) {
    unsafe {
        if !b.mapped.is_null() {
            (f.vkUnmapMemory)(dev, b.memory);
            b.mapped = ptr::null_mut();
        }
        if b.handle != VK_NULL_HANDLE {
            (f.vkDestroyBuffer)(dev, b.handle, ptr::null());
            b.handle = VK_NULL_HANDLE;
        }
        if b.memory != VK_NULL_HANDLE {
            (f.vkFreeMemory)(dev, b.memory, ptr::null());
            b.memory = VK_NULL_HANDLE;
        }
        b.capacity = 0;
    }
}

fn grow_host_buffer(
    f: &VkDeviceFns,
    dev: VkDevice,
    mem: &VkPhysicalDeviceMemoryProperties,
    b: &mut HostBuffer,
    needed: usize,
) -> Result<bool, String> {
    if needed <= b.capacity {
        return Ok(false);
    }
    let mut new_cap = b.capacity.max(4096);
    while new_cap < needed {
        new_cap = new_cap.saturating_mul(2);
    }
    destroy_host_buffer(f, dev, b);
    *b = create_host_buffer(f, dev, mem, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, new_cap)?;
    Ok(true)
}

fn upload_host_buffer(b: &HostBuffer, bytes: &[u8]) -> Result<(), String> {
    if bytes.len() > b.capacity {
        return Err("E_UI_VULKAN: host buffer upload exceeds capacity".to_string());
    }
    if !bytes.is_empty() {
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), b.mapped, bytes.len()) };
    }
    Ok(())
}

// ── Vulkan renderer ────────────────────────────────────────────────────────────────────────
pub struct VulkanRenderer {
    /// Kept alive for the lifetime of the renderer; dropping it unloads libvulkan.
    _lib: DynLib,
    _get_instance_proc: PFN_vkGetInstanceProcAddr,
    _global: VkGlobalFns,
    instance_fns: VkInstanceFns,
    instance: VkInstance,
    surface: VkSurfaceKHR,
    physical: VkPhysicalDevice,
    memory_props: VkPhysicalDeviceMemoryProperties,
    device_fns: VkDeviceFns,
    device: VkDevice,
    queue: VkQueue,
    queue_family: u32,

    swapchain: Option<Swapchain>,
    render_pass: VkRenderPass,
    descriptor_layout: VkDescriptorSetLayout,
    descriptor_pool: VkDescriptorPool,
    pipeline_layout: VkPipelineLayout,
    pipelines: [VkPipeline; 6],
    descriptor_sets: [[VkDescriptorSet; 6]; FRAMES_IN_FLIGHT],
    command_pool: VkCommandPool,
    command_buffers: [VkCommandBuffer; FRAMES_IN_FLIGHT],
    fences: [VkFence; FRAMES_IN_FLIGHT],
    image_available: [VkSemaphore; FRAMES_IN_FLIGHT],
    /// One render-finished semaphore per swapchain image. Reusing a per-frame semaphore would be
    /// invalid when the presentation engine has not consumed the previous signal yet.
    present_semaphores: Vec<VkSemaphore>,
    frames: [FrameResources; FRAMES_IN_FLIGHT],
    atlas: Option<Atlas>,
    frame: usize,
    extent: VkExtent2D,
    context_lost: bool,
    initialized: bool,
}

impl VulkanRenderer {
    /// Create the Vulkan instance/device/swapchain for `surface_info`.
    ///
    /// # Safety
    ///
    /// `surface_info` must hold a live native window handle of the matching `SurfaceKind` that
    /// stays valid for the lifetime of the returned renderer.
    pub unsafe fn new(surface_info: NativeSurface, atlas: &GlyphAtlas) -> Result<Self, String> {
        let lib =
            DynLib::open(&["libvulkan.so.1", "libvulkan.so", "vulkan-1.dll"]).ok_or_else(|| {
                "E_UI_VULKAN: no libvulkan.so.1 / vulkan-1.dll on this system".to_string()
            })?;
        let get_instance_proc_ptr = lib.sym("vkGetInstanceProcAddr");
        if get_instance_proc_ptr.is_null() {
            return Err("E_UI_VULKAN: vkGetInstanceProcAddr is missing".to_string());
        }
        let get_instance_proc: PFN_vkGetInstanceProcAddr =
            std::mem::transmute(get_instance_proc_ptr);

        let global = VkGlobalFns::load(&|name| get_instance_proc(ptr::null_mut(), name.as_ptr()))?;

        let required: &[&CStr] = match surface_info.kind {
            SurfaceKind::Wayland => &[
                VK_KHR_SURFACE_EXTENSION_NAME,
                VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
            ],
            SurfaceKind::Xlib => &[
                VK_KHR_SURFACE_EXTENSION_NAME,
                VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
            ],
            SurfaceKind::Win32 => &[
                VK_KHR_SURFACE_EXTENSION_NAME,
                VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
            ],
            SurfaceKind::Android => &[
                VK_KHR_SURFACE_EXTENSION_NAME,
                VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
            ],
        };
        Self::verify_instance_extensions(&global, required)?;

        let app_name = CString::new("Sofuu UI").unwrap();
        let engine_name = CString::new("sofuu-gpu").unwrap();
        let app = VkApplicationInfo {
            sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext: ptr::null(),
            pApplicationName: app_name.as_ptr(),
            applicationVersion: 1,
            pEngineName: engine_name.as_ptr(),
            engineVersion: 1,
            apiVersion: 0x0040_1000, // Vulkan 1.1
        };
        let ext_ptrs: Vec<*const c_char> = required.iter().map(|s| s.as_ptr()).collect();
        let instance_info = VkInstanceCreateInfo {
            sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            pApplicationInfo: &app,
            enabledLayerCount: 0,
            ppEnabledLayerNames: ptr::null(),
            enabledExtensionCount: ext_ptrs.len() as u32,
            ppEnabledExtensionNames: ext_ptrs.as_ptr(),
        };
        let mut instance: VkInstance = ptr::null_mut();
        vk_check(
            (global.vkCreateInstance)(&instance_info, ptr::null(), &mut instance),
            "vkCreateInstance",
        )?;

        let instance_fns = VkInstanceFns::load(&|name| get_instance_proc(instance, name.as_ptr()))?;

        // Platform surface creation entry points are only returned by vkGetInstanceProcAddr when
        // that platform's surface extension is enabled; resolve just the one we requested.
        let surface = match Self::create_surface(get_instance_proc, instance, surface_info) {
            Ok(s) => s,
            Err(err) => {
                (instance_fns.vkDestroyInstance)(instance, ptr::null());
                return Err(err);
            }
        };

        let (physical, queue_family) =
            match Self::pick_physical_device(&instance_fns, instance, surface) {
                Ok(v) => v,
                Err(err) => {
                    (instance_fns.vkDestroySurfaceKHR)(instance, surface, ptr::null());
                    (instance_fns.vkDestroyInstance)(instance, ptr::null());
                    return Err(err);
                }
            };
        let mut memory_props: VkPhysicalDeviceMemoryProperties = std::mem::zeroed();
        (instance_fns.vkGetPhysicalDeviceMemoryProperties)(physical, &mut memory_props);

        let priority = 1.0f32;
        let queue_info = VkDeviceQueueCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            queueFamilyIndex: queue_family,
            queueCount: 1,
            pQueuePriorities: &priority,
        };
        let device_ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME.as_ptr();
        let device_info = VkDeviceCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            queueCreateInfoCount: 1,
            pQueueCreateInfos: &queue_info,
            enabledLayerCount: 0,
            ppEnabledLayerNames: ptr::null(),
            enabledExtensionCount: 1,
            ppEnabledExtensionNames: &device_ext,
            pEnabledFeatures: ptr::null(),
        };
        let mut device: VkDevice = ptr::null_mut();
        if let Err(err) = vk_check(
            (instance_fns.vkCreateDevice)(physical, &device_info, ptr::null(), &mut device),
            "vkCreateDevice",
        ) {
            (instance_fns.vkDestroySurfaceKHR)(instance, surface, ptr::null());
            (instance_fns.vkDestroyInstance)(instance, ptr::null());
            return Err(err);
        }
        let mut queue: VkQueue = ptr::null_mut();
        let get_device_proc_ptr =
            (instance_fns.vkGetDeviceProcAddr)(device, c"vkGetDeviceProcAddr".as_ptr());
        let get_device_proc: PFN_vkGetDeviceProcAddr = if get_device_proc_ptr.is_null() {
            let p = get_instance_proc(instance, c"vkGetDeviceProcAddr".as_ptr());
            if p.is_null() {
                (instance_fns.vkDestroySurfaceKHR)(instance, surface, ptr::null());
                (instance_fns.vkDestroyInstance)(instance, ptr::null());
                return Err("E_UI_VULKAN: vkGetDeviceProcAddr is unavailable".to_string());
            }
            std::mem::transmute::<
                *const libc::c_void,
                unsafe extern "system" fn(
                    *mut libc::c_void,
                    *const libc::c_char,
                ) -> *const libc::c_void,
            >(p)
        } else {
            std::mem::transmute::<
                *const libc::c_void,
                unsafe extern "system" fn(
                    *mut libc::c_void,
                    *const libc::c_char,
                ) -> *const libc::c_void,
            >(get_device_proc_ptr)
        };
        let device_fns = match VkDeviceFns::load(&|name| {
            let p = get_device_proc(device, name.as_ptr());
            if p.is_null() {
                get_instance_proc(instance, name.as_ptr())
            } else {
                p
            }
        }) {
            Ok(f) => f,
            Err(err) => {
                (instance_fns.vkDestroySurfaceKHR)(instance, surface, ptr::null());
                (instance_fns.vkDestroyInstance)(instance, ptr::null());
                return Err(err);
            }
        };
        (device_fns.vkGetDeviceQueue)(device, queue_family, 0, &mut queue);

        let mut renderer = Self {
            _lib: lib,
            _get_instance_proc: get_instance_proc,
            _global: global,
            instance_fns,
            instance,
            surface,
            physical,
            memory_props,
            device_fns,
            device,
            queue,
            queue_family,
            swapchain: None,
            render_pass: VK_NULL_HANDLE,
            descriptor_layout: VK_NULL_HANDLE,
            descriptor_pool: VK_NULL_HANDLE,
            pipeline_layout: VK_NULL_HANDLE,
            pipelines: [VK_NULL_HANDLE; 6],
            descriptor_sets: [[VK_NULL_HANDLE; 6]; FRAMES_IN_FLIGHT],
            command_pool: VK_NULL_HANDLE,
            command_buffers: [ptr::null_mut(); FRAMES_IN_FLIGHT],
            fences: [VK_NULL_HANDLE; FRAMES_IN_FLIGHT],
            image_available: [VK_NULL_HANDLE; FRAMES_IN_FLIGHT],
            present_semaphores: Vec::new(),
            frames: std::array::from_fn(|_| FrameResources {
                quads: HostBuffer {
                    handle: 0,
                    memory: 0,
                    mapped: ptr::null_mut(),
                    capacity: 0,
                },
                ribbons: HostBuffer {
                    handle: 0,
                    memory: 0,
                    mapped: ptr::null_mut(),
                    capacity: 0,
                },
                strokes: HostBuffer {
                    handle: 0,
                    memory: 0,
                    mapped: ptr::null_mut(),
                    capacity: 0,
                },
                glyphs: HostBuffer {
                    handle: 0,
                    memory: 0,
                    mapped: ptr::null_mut(),
                    capacity: 0,
                },
                effects: HostBuffer {
                    handle: 0,
                    memory: 0,
                    mapped: ptr::null_mut(),
                    capacity: 0,
                },
                particles: HostBuffer {
                    handle: 0,
                    memory: 0,
                    mapped: ptr::null_mut(),
                    capacity: 0,
                },
                clips: HostBuffer {
                    handle: 0,
                    memory: 0,
                    mapped: ptr::null_mut(),
                    capacity: 0,
                },
            }),
            atlas: None,
            frame: 0,
            extent: VkExtent2D {
                width: 1,
                height: 1,
            },
            context_lost: false,
            initialized: false,
        };

        renderer.initialize_objects(atlas)?;
        renderer.initialized = true;
        Ok(renderer)
    }

    unsafe fn verify_instance_extensions(
        global: &VkGlobalFns,
        required: &[&CStr],
    ) -> Result<(), String> {
        let mut count = 0u32;
        (global.vkEnumerateInstanceExtensionProperties)(ptr::null(), &mut count, ptr::null_mut());
        let mut props = vec![std::mem::zeroed::<VkExtensionProperties>(); count as usize];
        if count > 0 {
            (global.vkEnumerateInstanceExtensionProperties)(
                ptr::null(),
                &mut count,
                props.as_mut_ptr(),
            );
        }
        for want in required {
            let found = props.iter().take(count as usize).any(|p| {
                let name = unsafe { CStr::from_ptr(p.extensionName.as_ptr()) };
                name == *want
            });
            if !found {
                return Err(format!(
                    "E_UI_VULKAN: required instance extension {} is unavailable",
                    want.to_string_lossy()
                ));
            }
        }
        Ok(())
    }

    unsafe fn pick_physical_device(
        f: &VkInstanceFns,
        instance: VkInstance,
        surface: VkSurfaceKHR,
    ) -> Result<(VkPhysicalDevice, u32), String> {
        let mut count = 0u32;
        vk_check(
            (f.vkEnumeratePhysicalDevices)(instance, &mut count, ptr::null_mut()),
            "vkEnumeratePhysicalDevices",
        )?;
        if count == 0 {
            return Err("E_UI_VULKAN: no Vulkan physical devices".to_string());
        }
        let mut devices = vec![ptr::null_mut(); count as usize];
        vk_check(
            (f.vkEnumeratePhysicalDevices)(instance, &mut count, devices.as_mut_ptr()),
            "vkEnumeratePhysicalDevices",
        )?;
        for device in devices.into_iter().take(count as usize) {
            let mut families = 0u32;
            (f.vkGetPhysicalDeviceQueueFamilyProperties)(device, &mut families, ptr::null_mut());
            if families == 0 {
                continue;
            }
            let mut props = vec![std::mem::zeroed::<VkQueueFamilyProperties>(); families as usize];
            (f.vkGetPhysicalDeviceQueueFamilyProperties)(device, &mut families, props.as_mut_ptr());
            for (idx, p) in props.iter().take(families as usize).enumerate() {
                if p.queueFlags & VK_QUEUE_GRAPHICS_BIT == 0 || p.queueCount == 0 {
                    continue;
                }
                let mut supported = 0u32;
                let r = (f.vkGetPhysicalDeviceSurfaceSupportKHR)(
                    device,
                    idx as u32,
                    surface,
                    &mut supported,
                );
                if r == VK_SUCCESS && supported != 0 {
                    return Ok((device, idx as u32));
                }
            }
        }
        Err("E_UI_VULKAN: no graphics queue family supports this surface".to_string())
    }

    unsafe fn create_surface(
        get_instance_proc: PFN_vkGetInstanceProcAddr,
        instance: VkInstance,
        surface_info: NativeSurface,
    ) -> Result<VkSurfaceKHR, String> {
        unsafe fn proc<T>(
            get: PFN_vkGetInstanceProcAddr,
            instance: VkInstance,
            name: &[u8],
        ) -> Result<T, String> {
            let p = get(instance, name.as_ptr().cast());
            if p.is_null() {
                let label =
                    std::str::from_utf8(&name[..name.len().saturating_sub(1)]).unwrap_or("unknown");
                return Err(format!(
                    "E_UI_VULKAN_MISSING: required entry point `{label}` is unavailable"
                ));
            }
            Ok(std::mem::transmute_copy(&p))
        }

        let mut surface: VkSurfaceKHR = VK_NULL_HANDLE;
        let result = match surface_info.kind {
            SurfaceKind::Wayland => {
                let create: PFN_vkCreateWaylandSurfaceKHR =
                    proc(get_instance_proc, instance, b"vkCreateWaylandSurfaceKHR\0")?;
                let info = VkWaylandSurfaceCreateInfoKHR {
                    sType: VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
                    pNext: ptr::null(),
                    flags: 0,
                    display: surface_info.display,
                    surface: surface_info.window,
                };
                create(instance, &info, ptr::null(), &mut surface)
            }
            SurfaceKind::Xlib => {
                let create: PFN_vkCreateXlibSurfaceKHR =
                    proc(get_instance_proc, instance, b"vkCreateXlibSurfaceKHR\0")?;
                let info = VkXlibSurfaceCreateInfoKHR {
                    sType: VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                    pNext: ptr::null(),
                    flags: 0,
                    dpy: surface_info.display,
                    window: surface_info.window as usize,
                };
                create(instance, &info, ptr::null(), &mut surface)
            }
            SurfaceKind::Win32 => {
                let create: PFN_vkCreateWin32SurfaceKHR =
                    proc(get_instance_proc, instance, b"vkCreateWin32SurfaceKHR\0")?;
                let info = VkWin32SurfaceCreateInfoKHR {
                    sType: VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                    pNext: ptr::null(),
                    flags: 0,
                    hinstance: surface_info.display,
                    hwnd: surface_info.window,
                };
                create(instance, &info, ptr::null(), &mut surface)
            }
            SurfaceKind::Android => {
                let create: PFN_vkCreateAndroidSurfaceKHR =
                    proc(get_instance_proc, instance, b"vkCreateAndroidSurfaceKHR\0")?;
                let info = VkAndroidSurfaceCreateInfoKHR {
                    sType: VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
                    pNext: ptr::null(),
                    flags: 0,
                    window: surface_info.window,
                };
                create(instance, &info, ptr::null(), &mut surface)
            }
        };
        vk_check(result, "vkCreate*SurfaceKHR")?;
        if surface == VK_NULL_HANDLE {
            return Err("E_UI_VULKAN: vkCreate*SurfaceKHR returned VK_NULL_HANDLE".to_string());
        }
        Ok(surface)
    }

    unsafe fn initialize_objects(&mut self, atlas: &GlyphAtlas) -> Result<(), String> {
        let f = self.device_fns;
        let dev = self.device;
        let pool_info = VkCommandPoolCreateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext: ptr::null(),
            flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex: self.queue_family,
        };
        vk_check(
            (f.vkCreateCommandPool)(dev, &pool_info, ptr::null(), &mut self.command_pool),
            "vkCreateCommandPool",
        )?;
        let alloc_info = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: ptr::null(),
            commandPool: self.command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: FRAMES_IN_FLIGHT as u32,
        };
        let mut cbs = [ptr::null_mut(); FRAMES_IN_FLIGHT];
        vk_check(
            (f.vkAllocateCommandBuffers)(dev, &alloc_info, cbs.as_mut_ptr()),
            "vkAllocateCommandBuffers",
        )?;
        self.command_buffers = cbs;

        self.reset_frame_sync()?;

        let mut frames: [FrameResources; FRAMES_IN_FLIGHT] =
            std::array::from_fn(|_| FrameResources::empty());
        for frame in &mut frames {
            let storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            frame.quads = create_host_buffer(&f, dev, &self.memory_props, storage, 256 * 1024)?;
            frame.ribbons = create_host_buffer(&f, dev, &self.memory_props, storage, 128 * 1024)?;
            frame.strokes = create_host_buffer(&f, dev, &self.memory_props, storage, 128 * 1024)?;
            frame.glyphs = create_host_buffer(&f, dev, &self.memory_props, storage, 256 * 1024)?;
            frame.effects = create_host_buffer(&f, dev, &self.memory_props, storage, 128 * 1024)?;
            frame.particles = create_host_buffer(&f, dev, &self.memory_props, storage, 64 * 1024)?;
            frame.clips = create_host_buffer(&f, dev, &self.memory_props, storage, 64 * 1024)?;
        }
        self.frames = frames;

        self.create_descriptor_layout()?;
        self.create_swapchain(1024, 680)?;
        self.create_render_pass()?;
        self.create_framebuffers()?;
        self.create_pipelines()?;
        self.atlas = Some(self.create_atlas(atlas)?);
        self.allocate_descriptor_sets()?;
        self.write_descriptors();
        Ok(())
    }

    unsafe fn create_descriptor_layout(&mut self) -> Result<(), String> {
        let bindings = [
            VkDescriptorSetLayoutBinding {
                binding: 0,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: 1,
                stageFlags: VK_SHADER_STAGE_VERTEX_BIT,
                pImmutableSamplers: ptr::null(),
            },
            VkDescriptorSetLayoutBinding {
                binding: 1,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: 1,
                // Declared in both stages; only the fragment stage reads it.
                stageFlags: VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                pImmutableSamplers: ptr::null(),
            },
            VkDescriptorSetLayoutBinding {
                binding: 2,
                descriptorType: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount: 1,
                stageFlags: VK_SHADER_STAGE_FRAGMENT_BIT,
                pImmutableSamplers: ptr::null(),
            },
        ];
        let info = VkDescriptorSetLayoutCreateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            bindingCount: 3,
            pBindings: bindings.as_ptr(),
        };
        vk_check(
            (self.device_fns.vkCreateDescriptorSetLayout)(
                self.device,
                &info,
                ptr::null(),
                &mut self.descriptor_layout,
            ),
            "vkCreateDescriptorSetLayout",
        )?;

        let pool_sizes = [
            VkDescriptorPoolSize {
                ty: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: (FRAMES_IN_FLIGHT * 6 * 2) as u32,
            },
            VkDescriptorPoolSize {
                ty: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount: (FRAMES_IN_FLIGHT * 6) as u32,
            },
        ];
        let pool_info = VkDescriptorPoolCreateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            maxSets: (FRAMES_IN_FLIGHT * 6) as u32,
            poolSizeCount: pool_sizes.len() as u32,
            pPoolSizes: pool_sizes.as_ptr(),
        };
        vk_check(
            (self.device_fns.vkCreateDescriptorPool)(
                self.device,
                &pool_info,
                ptr::null(),
                &mut self.descriptor_pool,
            ),
            "vkCreateDescriptorPool",
        )
    }

    unsafe fn allocate_descriptor_sets(&mut self) -> Result<(), String> {
        let layouts = [self.descriptor_layout; FRAMES_IN_FLIGHT * 6];
        let info = VkDescriptorSetAllocateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            pNext: ptr::null(),
            descriptorPool: self.descriptor_pool,
            descriptorSetCount: layouts.len() as u32,
            pSetLayouts: layouts.as_ptr(),
        };
        let mut sets = [VK_NULL_HANDLE; FRAMES_IN_FLIGHT * 6];
        vk_check(
            (self.device_fns.vkAllocateDescriptorSets)(self.device, &info, sets.as_mut_ptr()),
            "vkAllocateDescriptorSets",
        )?;
        for f in 0..FRAMES_IN_FLIGHT {
            for k in 0..6 {
                self.descriptor_sets[f][k] = sets[f * 6 + k];
            }
        }
        Ok(())
    }

    unsafe fn write_descriptors(&mut self) {
        let atlas = match &self.atlas {
            Some(a) => a,
            None => return,
        };
        let mut buffer_infos: Vec<VkDescriptorBufferInfo> =
            Vec::with_capacity(FRAMES_IN_FLIGHT * 7 * 2);
        for frame in 0..FRAMES_IN_FLIGHT {
            let f = &self.frames[frame];
            buffer_infos.push(VkDescriptorBufferInfo {
                buffer: f.quads.handle,
                offset: 0,
                range: VK_WHOLE_SIZE,
            });
            buffer_infos.push(VkDescriptorBufferInfo {
                buffer: f.ribbons.handle,
                offset: 0,
                range: VK_WHOLE_SIZE,
            });
            buffer_infos.push(VkDescriptorBufferInfo {
                buffer: f.strokes.handle,
                offset: 0,
                range: VK_WHOLE_SIZE,
            });
            buffer_infos.push(VkDescriptorBufferInfo {
                buffer: f.glyphs.handle,
                offset: 0,
                range: VK_WHOLE_SIZE,
            });
            buffer_infos.push(VkDescriptorBufferInfo {
                buffer: f.effects.handle,
                offset: 0,
                range: VK_WHOLE_SIZE,
            });
            buffer_infos.push(VkDescriptorBufferInfo {
                buffer: f.particles.handle,
                offset: 0,
                range: VK_WHOLE_SIZE,
            });
            buffer_infos.push(VkDescriptorBufferInfo {
                buffer: f.clips.handle,
                offset: 0,
                range: VK_WHOLE_SIZE,
            });
        }
        let image_info = VkDescriptorImageInfo {
            sampler: atlas.sampler,
            imageView: atlas.view,
            imageLayout: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        let mut writes: Vec<VkWriteDescriptorSet> = Vec::with_capacity(FRAMES_IN_FLIGHT * 6 * 3);
        for frame in 0..FRAMES_IN_FLIGHT {
            let instance_infos = &buffer_infos[frame * 7..frame * 7 + 6];
            for (kind, instance_info) in instance_infos.iter().enumerate() {
                let set = self.descriptor_sets[frame][kind];
                writes.push(VkWriteDescriptorSet {
                    sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    pNext: ptr::null(),
                    dstSet: set,
                    dstBinding: 0,
                    dstArrayElement: 0,
                    descriptorCount: 1,
                    descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pImageInfo: ptr::null(),
                    pBufferInfo: instance_info,
                    pTexelBufferView: ptr::null(),
                });
                writes.push(VkWriteDescriptorSet {
                    sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    pNext: ptr::null(),
                    dstSet: set,
                    dstBinding: 1,
                    dstArrayElement: 0,
                    descriptorCount: 1,
                    descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pImageInfo: ptr::null(),
                    pBufferInfo: &buffer_infos[frame * 7 + 6],
                    pTexelBufferView: ptr::null(),
                });
                writes.push(VkWriteDescriptorSet {
                    sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    pNext: ptr::null(),
                    dstSet: set,
                    dstBinding: 2,
                    dstArrayElement: 0,
                    descriptorCount: 1,
                    descriptorType: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo: &image_info,
                    pBufferInfo: ptr::null(),
                    pTexelBufferView: ptr::null(),
                });
            }
        }
        (self.device_fns.vkUpdateDescriptorSets)(
            self.device,
            writes.len() as u32,
            writes.as_ptr(),
            0,
            ptr::null(),
        );
    }

    unsafe fn create_swapchain(&mut self, width: u32, height: u32) -> Result<(), String> {
        let f = self.device_fns;
        let dev = self.device;
        let mut caps: VkSurfaceCapabilitiesKHR = std::mem::zeroed();
        vk_check(
            (self.instance_fns.vkGetPhysicalDeviceSurfaceCapabilitiesKHR)(
                self.physical,
                self.surface,
                &mut caps,
            ),
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
        )?;
        let mut format_count = 0u32;
        (self.instance_fns.vkGetPhysicalDeviceSurfaceFormatsKHR)(
            self.physical,
            self.surface,
            &mut format_count,
            ptr::null_mut(),
        );
        let mut formats = vec![std::mem::zeroed::<VkSurfaceFormatKHR>(); format_count as usize];
        if format_count > 0 {
            (self.instance_fns.vkGetPhysicalDeviceSurfaceFormatsKHR)(
                self.physical,
                self.surface,
                &mut format_count,
                formats.as_mut_ptr(),
            );
        }
        let format = formats
            .iter()
            .take(format_count as usize)
            .find(|f| {
                f.format == VK_FORMAT_B8G8R8A8_UNORM
                    && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
            })
            .or_else(|| {
                formats
                    .iter()
                    .take(format_count as usize)
                    .find(|f| f.format == VK_FORMAT_R8G8B8A8_UNORM)
            })
            .or_else(|| formats.iter().take(format_count as usize).next())
            .copied()
            .ok_or_else(|| "E_UI_VULKAN: the surface reports no formats".to_string())?;
        let extent = if caps.currentExtent.width != u32::MAX {
            caps.currentExtent
        } else {
            VkExtent2D {
                width: width.clamp(caps.minImageExtent.width, caps.maxImageExtent.width),
                height: height.clamp(caps.minImageExtent.height, caps.maxImageExtent.height),
            }
        };
        let mut image_count = caps.minImageCount + 1;
        if caps.maxImageCount > 0 && image_count > caps.maxImageCount {
            image_count = caps.maxImageCount;
        }
        // Sofuu renders a transparent, premultiplied-alpha surface; asking for OPAQUE would
        // discard the alpha channel and turn the window into a black rectangle on compositors.
        let composite =
            if caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR != 0 {
                VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR
            } else if caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR != 0 {
                VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
            } else {
                VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
            };
        let info = VkSwapchainCreateInfoKHR {
            sType: VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            pNext: ptr::null(),
            flags: 0,
            surface: self.surface,
            minImageCount: image_count,
            imageFormat: format.format,
            imageColorSpace: format.colorSpace,
            imageExtent: extent,
            imageArrayLayers: 1,
            imageUsage: 0x10, // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
            imageSharingMode: VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
            preTransform: caps.currentTransform,
            compositeAlpha: composite,
            presentMode: VK_PRESENT_MODE_FIFO_KHR,
            clipped: 1,
            oldSwapchain: VK_NULL_HANDLE,
        };
        let mut handle: VkSwapchainKHR = VK_NULL_HANDLE;
        vk_check(
            (f.vkCreateSwapchainKHR)(dev, &info, ptr::null(), &mut handle),
            "vkCreateSwapchainKHR",
        )?;
        let mut count = 0u32;
        (f.vkGetSwapchainImagesKHR)(dev, handle, &mut count, ptr::null_mut());
        let mut images = vec![VK_NULL_HANDLE; count as usize];
        vk_check(
            (f.vkGetSwapchainImagesKHR)(dev, handle, &mut count, images.as_mut_ptr()),
            "vkGetSwapchainImagesKHR",
        )?;
        let mut views = Vec::with_capacity(images.len());
        for image in &images {
            let view_info = VkImageViewCreateInfo {
                sType: VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                image: *image,
                viewType: VK_IMAGE_VIEW_TYPE_2D,
                format: format.format,
                components: [0, 1, 2, 3],
                subresourceRange: VkImageSubresourceRange {
                    aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel: 0,
                    levelCount: 1,
                    baseArrayLayer: 0,
                    layerCount: 1,
                },
            };
            let mut view = VK_NULL_HANDLE;
            vk_check(
                (f.vkCreateImageView)(dev, &view_info, ptr::null(), &mut view),
                "vkCreateImageView",
            )?;
            views.push(view);
        }
        let sem_info = VkSemaphoreCreateInfo {
            sType: VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
        };
        let mut present_semaphores = Vec::with_capacity(views.len());
        for _ in 0..views.len() {
            let mut sem = VK_NULL_HANDLE;
            vk_check(
                (f.vkCreateSemaphore)(dev, &sem_info, ptr::null(), &mut sem),
                "vkCreateSemaphore(present)",
            )?;
            present_semaphores.push(sem);
        }
        self.present_semaphores = present_semaphores;
        self.swapchain = Some(Swapchain {
            handle,
            _images: images,
            views,
            framebuffers: Vec::new(),
            extent,
            format: format.format,
        });
        self.extent = extent;
        Ok(())
    }

    /// Recreate the per-frame acquire semaphores and fences. Any frame that failed after
    /// `vkAcquireNextImageKHR` leaves its semaphore signalled; recreating them during a
    /// swapchain rebuild prevents the next acquire from reusing a signalled semaphore.
    unsafe fn reset_frame_sync(&mut self) -> Result<(), String> {
        let f = self.device_fns;
        let dev = self.device;
        for i in 0..FRAMES_IN_FLIGHT {
            if self.fences[i] != VK_NULL_HANDLE {
                (f.vkDestroyFence)(dev, self.fences[i], ptr::null());
                self.fences[i] = VK_NULL_HANDLE;
            }
            if self.image_available[i] != VK_NULL_HANDLE {
                (f.vkDestroySemaphore)(dev, self.image_available[i], ptr::null());
                self.image_available[i] = VK_NULL_HANDLE;
            }
        }
        let sem = VkSemaphoreCreateInfo {
            sType: VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
        };
        let fence = VkFenceCreateInfo {
            sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            pNext: ptr::null(),
            flags: VK_FENCE_CREATE_SIGNALED_BIT,
        };
        for i in 0..FRAMES_IN_FLIGHT {
            vk_check(
                (f.vkCreateSemaphore)(dev, &sem, ptr::null(), &mut self.image_available[i]),
                "vkCreateSemaphore",
            )?;
            vk_check(
                (f.vkCreateFence)(dev, &fence, ptr::null(), &mut self.fences[i]),
                "vkCreateFence",
            )?;
        }
        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self) {
        let f = self.device_fns;
        for sem in self.present_semaphores.drain(..) {
            (f.vkDestroySemaphore)(self.device, sem, ptr::null());
        }
        if let Some(mut sc) = self.swapchain.take() {
            for fb in sc.framebuffers.drain(..) {
                (f.vkDestroyFramebuffer)(self.device, fb, ptr::null());
            }
            for view in sc.views.drain(..) {
                (f.vkDestroyImageView)(self.device, view, ptr::null());
            }
            (f.vkDestroySwapchainKHR)(self.device, sc.handle, ptr::null());
        }
    }

    unsafe fn create_render_pass(&mut self) -> Result<(), String> {
        let format = self.swapchain.as_ref().unwrap().format;
        let attachment = VkAttachmentDescription {
            flags: 0,
            format,
            samples: VK_SAMPLE_COUNT_1_BIT,
            loadOp: VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp: VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp: VK_ATTACHMENT_LOAD_OP_CLEAR,
            stencilStoreOp: VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout: VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout: VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };
        let color_ref = VkAttachmentReference {
            attachment: 0,
            layout: 2,
        };
        let subpass = VkSubpassDescription {
            flags: 0,
            pipelineBindPoint: VK_PIPELINE_BIND_POINT_GRAPHICS,
            inputAttachmentCount: 0,
            pInputAttachments: ptr::null(),
            colorAttachmentCount: 1,
            pColorAttachments: &color_ref,
            pResolveAttachments: ptr::null(),
            pDepthStencilAttachment: ptr::null(),
            preserveAttachmentCount: 0,
            pPreserveAttachments: ptr::null(),
        };
        let dependency = VkSubpassDependency {
            srcSubpass: VK_SUBPASS_EXTERNAL,
            dstSubpass: 0,
            srcStageMask: VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstStageMask: VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask: 0,
            dstAccessMask: VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dependencyFlags: 0,
        };
        let info = VkRenderPassCreateInfo {
            sType: VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            attachmentCount: 1,
            pAttachments: &attachment,
            subpassCount: 1,
            pSubpasses: &subpass,
            dependencyCount: 1,
            pDependencies: &dependency,
        };
        vk_check(
            (self.device_fns.vkCreateRenderPass)(
                self.device,
                &info,
                ptr::null(),
                &mut self.render_pass,
            ),
            "vkCreateRenderPass",
        )
    }

    unsafe fn create_framebuffers(&mut self) -> Result<(), String> {
        let f = self.device_fns;
        let sc = self.swapchain.as_mut().unwrap();
        sc.framebuffers.clear();
        for view in &sc.views {
            let info = VkFramebufferCreateInfo {
                sType: VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                renderPass: self.render_pass,
                attachmentCount: 1,
                pAttachments: view,
                width: sc.extent.width,
                height: sc.extent.height,
                layers: 1,
            };
            let mut fb = VK_NULL_HANDLE;
            vk_check(
                (f.vkCreateFramebuffer)(self.device, &info, ptr::null(), &mut fb),
                "vkCreateFramebuffer",
            )?;
            sc.framebuffers.push(fb);
        }
        Ok(())
    }

    unsafe fn create_pipelines(&mut self) -> Result<(), String> {
        let f = self.device_fns;
        let dev = self.device;
        let push = VkPushConstantRange {
            stageFlags: VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        };
        let layout_info = VkPipelineLayoutCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            setLayoutCount: 1,
            pSetLayouts: &self.descriptor_layout,
            pushConstantRangeCount: 1,
            pPushConstantRanges: &push,
        };
        vk_check(
            (f.vkCreatePipelineLayout)(dev, &layout_info, ptr::null(), &mut self.pipeline_layout),
            "vkCreatePipelineLayout",
        )?;

        let dynamic_states = [VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR];
        let dynamic = VkPipelineDynamicStateCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            dynamicStateCount: dynamic_states.len() as u32,
            pDynamicStates: dynamic_states.as_ptr(),
        };
        let vertex_input = VkPipelineVertexInputStateCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            vertexBindingDescriptionCount: 0,
            pVertexBindingDescriptions: ptr::null(),
            vertexAttributeDescriptionCount: 0,
            pVertexAttributeDescriptions: ptr::null(),
        };
        let input_assembly = VkPipelineInputAssemblyStateCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            topology: VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
            primitiveRestartEnable: 0,
        };
        let viewport_state = VkPipelineViewportStateCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            viewportCount: 1,
            pViewports: ptr::null(),
            scissorCount: 1,
            pScissors: ptr::null(),
        };
        let raster = VkPipelineRasterizationStateCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            depthClampEnable: 0,
            rasterizerDiscardEnable: 0,
            polygonMode: VK_POLYGON_MODE_FILL,
            cullMode: VK_CULL_MODE_NONE,
            frontFace: VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthBiasEnable: 0,
            depthBiasConstantFactor: 0.0,
            depthBiasClamp: 0.0,
            depthBiasSlopeFactor: 0.0,
            lineWidth: 1.0,
        };
        let multisample = VkPipelineMultisampleStateCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            rasterizationSamples: VK_SAMPLE_COUNT_1_BIT,
            sampleShadingEnable: 0,
            minSampleShading: 1.0,
            pSampleMask: ptr::null(),
            alphaToCoverageEnable: 0,
            alphaToOneEnable: 0,
        };
        let attachment_blend = VkPipelineColorBlendAttachmentState {
            blendEnable: 1,
            srcColorBlendFactor: VK_BLEND_FACTOR_ONE,
            dstColorBlendFactor: VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            colorBlendOp: VK_BLEND_OP_ADD,
            srcAlphaBlendFactor: VK_BLEND_FACTOR_ONE,
            dstAlphaBlendFactor: VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            alphaBlendOp: VK_BLEND_OP_ADD,
            colorWriteMask: VK_COLOR_COMPONENT_RGBA_BIT,
        };
        let blend = VkPipelineColorBlendStateCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            logicOpEnable: 0,
            logicOp: 0,
            attachmentCount: 1,
            pAttachments: &attachment_blend,
            blendConstants: [0.0; 4],
        };

        let entry = CString::new("main").unwrap();
        for (slot, (vert, frag)) in [
            (0usize, (SHADER_QUAD_VERT, SHADER_QUAD_FRAG)),
            (1, (SHADER_RIBBON_VERT, SHADER_RIBBON_FRAG)),
            (2, (SHADER_STROKE_VERT, SHADER_STROKE_FRAG)),
            (3, (SHADER_GLYPH_VERT, SHADER_GLYPH_FRAG)),
            (4, (SHADER_EFFECT_VERT, SHADER_EFFECT_FRAG)),
            (5, (SHADER_PARTICLE_VERT, SHADER_PARTICLE_FRAG)),
        ] {
            let vert_module = self.create_shader_module(vert)?;
            let frag_module = self.create_shader_module(frag)?;
            let stages = [
                VkPipelineShaderStageCreateInfo {
                    sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0,
                    stage: VK_SHADER_STAGE_VERTEX_BIT,
                    module: vert_module,
                    pName: entry.as_ptr(),
                    pSpecializationInfo: ptr::null(),
                },
                VkPipelineShaderStageCreateInfo {
                    sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0,
                    stage: VK_SHADER_STAGE_FRAGMENT_BIT,
                    module: frag_module,
                    pName: entry.as_ptr(),
                    pSpecializationInfo: ptr::null(),
                },
            ];
            let info = VkGraphicsPipelineCreateInfo {
                sType: VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                stageCount: 2,
                pStages: stages.as_ptr(),
                pVertexInputState: &vertex_input,
                pInputAssemblyState: &input_assembly,
                pTessellationState: ptr::null(),
                pViewportState: &viewport_state,
                pRasterizationState: &raster,
                pMultisampleState: &multisample,
                pDepthStencilState: ptr::null(),
                pColorBlendState: &blend,
                pDynamicState: &dynamic,
                layout: self.pipeline_layout,
                renderPass: self.render_pass,
                subpass: 0,
                basePipelineHandle: VK_NULL_HANDLE,
                basePipelineIndex: -1,
            };
            let result = (f.vkCreateGraphicsPipelines)(
                dev,
                VK_NULL_HANDLE,
                1,
                &info,
                ptr::null(),
                &mut self.pipelines[slot],
            );
            (f.vkDestroyShaderModule)(dev, vert_module, ptr::null());
            (f.vkDestroyShaderModule)(dev, frag_module, ptr::null());
            vk_check(result, "vkCreateGraphicsPipelines")?;
        }
        Ok(())
    }

    unsafe fn create_shader_module(&self, bytes: &[u8]) -> Result<VkShaderModule, String> {
        if bytes.len() < 4 || !bytes.len().is_multiple_of(4) {
            return Err("E_UI_VULKAN: malformed embedded SPIR-V module".to_string());
        }
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let info = VkShaderModuleCreateInfo {
            sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            codeSize: bytes.len(),
            pCode: words.as_ptr(),
        };
        let mut module = VK_NULL_HANDLE;
        vk_check(
            (self.device_fns.vkCreateShaderModule)(self.device, &info, ptr::null(), &mut module),
            "vkCreateShaderModule",
        )?;
        Ok(module)
    }

    unsafe fn create_atlas(&self, atlas: &GlyphAtlas) -> Result<Atlas, String> {
        let f = self.device_fns;
        let dev = self.device;
        let image_info = VkImageCreateInfo {
            sType: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            imageType: VK_IMAGE_TYPE_2D,
            format: VK_FORMAT_R8_UNORM,
            extent: VkExtent3D {
                width: atlas.width.max(1),
                height: atlas.height.max(1),
                depth: 1,
            },
            mipLevels: 1,
            arrayLayers: 1,
            samples: VK_SAMPLE_COUNT_1_BIT,
            tiling: VK_IMAGE_TILING_OPTIMAL,
            usage: VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            sharingMode: VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
            initialLayout: VK_IMAGE_LAYOUT_UNDEFINED,
        };
        let mut image: VkImage = VK_NULL_HANDLE;
        vk_check(
            (f.vkCreateImage)(dev, &image_info, ptr::null(), &mut image),
            "vkCreateImage",
        )?;
        let mut req: VkMemoryRequirements = std::mem::zeroed();
        (f.vkGetImageMemoryRequirements)(dev, image, &mut req);
        let ty = find_memory_type(
            &self.memory_props,
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        .ok_or_else(|| {
            "E_UI_VULKAN: no device-local memory type for the glyph atlas".to_string()
        })?;
        let alloc = VkMemoryAllocateInfo {
            sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: ptr::null(),
            allocationSize: req.size,
            memoryTypeIndex: ty,
        };
        let mut memory: VkDeviceMemory = VK_NULL_HANDLE;
        vk_check(
            (f.vkAllocateMemory)(dev, &alloc, ptr::null(), &mut memory),
            "vkAllocateMemory",
        )?;
        vk_check(
            (f.vkBindImageMemory)(dev, image, memory, 0),
            "vkBindImageMemory",
        )?;

        let staging = create_host_buffer(
            &f,
            dev,
            &self.memory_props,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            atlas.pixels.len().max(16),
        )?;
        upload_host_buffer(&staging, &atlas.pixels)?;

        let cb_info = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: ptr::null(),
            commandPool: self.command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut cb: VkCommandBuffer = ptr::null_mut();
        vk_check(
            (f.vkAllocateCommandBuffers)(dev, &cb_info, &mut cb),
            "vkAllocateCommandBuffers",
        )?;
        let begin = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo: ptr::null(),
        };
        vk_check((f.vkBeginCommandBuffer)(cb, &begin), "vkBeginCommandBuffer")?;

        let range = VkImageSubresourceRange {
            aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel: 0,
            levelCount: 1,
            baseArrayLayer: 0,
            layerCount: 1,
        };
        let to_transfer = VkImageMemoryBarrier {
            sType: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            pNext: ptr::null(),
            srcAccessMask: 0,
            dstAccessMask: VK_ACCESS_TRANSFER_WRITE_BIT,
            oldLayout: VK_IMAGE_LAYOUT_UNDEFINED,
            newLayout: VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            srcQueueFamilyIndex: u32::MAX,
            dstQueueFamilyIndex: u32::MAX,
            image,
            subresourceRange: range,
        };
        (f.vkCmdPipelineBarrier)(
            cb,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            ptr::null(),
            0,
            ptr::null(),
            1,
            &to_transfer,
        );
        let region = VkBufferImageCopy {
            bufferOffset: 0,
            bufferRowLength: 0,
            bufferImageHeight: 0,
            imageSubresource: VkImageSubresourceLayers {
                aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel: 0,
                baseArrayLayer: 0,
                layerCount: 1,
            },
            imageOffset: VkOffset3D { x: 0, y: 0, z: 0 },
            imageExtent: VkExtent3D {
                width: atlas.width.max(1),
                height: atlas.height.max(1),
                depth: 1,
            },
        };
        (f.vkCmdCopyBufferToImage)(
            cb,
            staging.handle,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region,
        );
        let to_shader = VkImageMemoryBarrier {
            sType: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            pNext: ptr::null(),
            srcAccessMask: VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask: VK_ACCESS_SHADER_READ_BIT,
            oldLayout: VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            newLayout: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            srcQueueFamilyIndex: u32::MAX,
            dstQueueFamilyIndex: u32::MAX,
            image,
            subresourceRange: range,
        };
        (f.vkCmdPipelineBarrier)(
            cb,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0,
            ptr::null(),
            0,
            ptr::null(),
            1,
            &to_shader,
        );
        vk_check((f.vkEndCommandBuffer)(cb), "vkEndCommandBuffer")?;
        let submit = VkSubmitInfo {
            sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: ptr::null(),
            waitSemaphoreCount: 0,
            pWaitSemaphores: ptr::null(),
            pWaitDstStageMask: ptr::null(),
            commandBufferCount: 1,
            pCommandBuffers: &cb,
            signalSemaphoreCount: 0,
            pSignalSemaphores: ptr::null(),
        };
        vk_check(
            (f.vkQueueSubmit)(self.queue, 1, &submit, VK_NULL_HANDLE),
            "vkQueueSubmit(atlas)",
        )?;
        vk_check((f.vkQueueWaitIdle)(self.queue), "vkQueueWaitIdle(atlas)")?;
        let mut staging = staging;
        destroy_host_buffer(&f, dev, &mut staging);

        let view_info = VkImageViewCreateInfo {
            sType: VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            image,
            viewType: VK_IMAGE_VIEW_TYPE_2D,
            format: VK_FORMAT_R8_UNORM,
            components: [0, 1, 2, 3],
            subresourceRange: range,
        };
        let mut view: VkImageView = VK_NULL_HANDLE;
        vk_check(
            (f.vkCreateImageView)(dev, &view_info, ptr::null(), &mut view),
            "vkCreateImageView(atlas)",
        )?;
        let sampler_info = VkSamplerCreateInfo {
            sType: VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            magFilter: VK_FILTER_LINEAR,
            minFilter: VK_FILTER_LINEAR,
            mipmapMode: VK_SAMPLER_MIPMAP_MODE_NEAREST,
            addressModeU: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeV: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeW: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            mipLodBias: 0.0,
            anisotropyEnable: 0,
            maxAnisotropy: 1.0,
            compareEnable: 0,
            compareOp: VK_COMPARE_OP_NEVER,
            minLod: 0.0,
            maxLod: 0.0,
            borderColor: VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
            unnormalizedCoordinates: 0,
        };
        let mut sampler: VkSampler = VK_NULL_HANDLE;
        vk_check(
            (f.vkCreateSampler)(dev, &sampler_info, ptr::null(), &mut sampler),
            "vkCreateSampler",
        )?;
        Ok(Atlas {
            image,
            memory,
            view,
            sampler,
            width: atlas.width.max(1),
            height: atlas.height.max(1),
        })
    }

    pub fn context_lost(&self) -> bool {
        self.context_lost
    }

    pub fn recover_context(&mut self) {
        let width = self.extent.width.max(1);
        let height = self.extent.height.max(1);
        // Keep `context_lost` set on failure: the next frame retries instead of rendering into
        // a destroyed swapchain.
        self.context_lost = unsafe { self.rebuild_swapchain(width, height) }.is_err();
    }

    /// Replace the platform surface (Android `ANativeWindow` rotation / surface recreation)
    /// while preserving device, buffers, atlas and descriptor sets.
    ///
    /// # Safety
    ///
    /// `surface_info` must hold a live native window handle of the matching `SurfaceKind`.
    pub unsafe fn recreate_surface(&mut self, surface_info: NativeSurface) -> Result<(), String> {
        if surface_info.window.is_null() {
            // Android backgrounded: keep the old objects until a real window returns instead of
            // destroying the swapchain for a NULL ANativeWindow.
            return Err("E_UI_VULKAN: recreate_surface called without a native window".to_string());
        }
        vk_check(
            (self.device_fns.vkDeviceWaitIdle)(self.device),
            "vkDeviceWaitIdle(recreate surface)",
        )?;
        self.destroy_swapchain();
        self.reset_frame_sync()?;
        if self.render_pass != VK_NULL_HANDLE {
            (self.device_fns.vkDestroyRenderPass)(self.device, self.render_pass, ptr::null());
            self.render_pass = VK_NULL_HANDLE;
        }
        for p in &mut self.pipelines {
            if *p != VK_NULL_HANDLE {
                (self.device_fns.vkDestroyPipeline)(self.device, *p, ptr::null());
                *p = VK_NULL_HANDLE;
            }
        }
        if self.surface != VK_NULL_HANDLE {
            (self.instance_fns.vkDestroySurfaceKHR)(self.instance, self.surface, ptr::null());
            self.surface = VK_NULL_HANDLE;
        }
        self.context_lost = true;
        self.surface = Self::create_surface(self._get_instance_proc, self.instance, surface_info)?;
        let mut supported = 0u32;
        vk_check(
            (self.instance_fns.vkGetPhysicalDeviceSurfaceSupportKHR)(
                self.physical,
                self.queue_family,
                self.surface,
                &mut supported,
            ),
            "vkGetPhysicalDeviceSurfaceSupportKHR(recreated)",
        )?;
        if supported == 0 {
            return Err("E_UI_VULKAN: recreated surface has no present support".to_string());
        }
        let width = self.extent.width.max(1);
        let height = self.extent.height.max(1);
        self.create_swapchain(width, height)?;
        self.create_render_pass()?;
        self.create_framebuffers()?;
        self.create_pipelines()?;
        self.context_lost = false;
        Ok(())
    }

    unsafe fn rebuild_swapchain(&mut self, width: u32, height: u32) -> Result<(), String> {
        if self.surface == VK_NULL_HANDLE {
            return Err(
                "E_UI_VULKAN: platform surface was lost and could not be recreated".to_string(),
            );
        }
        let _ = (self.device_fns.vkDeviceWaitIdle)(self.device);
        self.destroy_swapchain();
        self.reset_frame_sync()?;
        if self.render_pass != VK_NULL_HANDLE {
            (self.device_fns.vkDestroyRenderPass)(self.device, self.render_pass, ptr::null());
            self.render_pass = VK_NULL_HANDLE;
        }
        for p in &mut self.pipelines {
            if *p != VK_NULL_HANDLE {
                (self.device_fns.vkDestroyPipeline)(self.device, *p, ptr::null());
                *p = VK_NULL_HANDLE;
            }
        }
        self.create_swapchain(width, height)?;
        self.create_render_pass()?;
        self.create_framebuffers()?;
        self.create_pipelines()?;
        Ok(())
    }

    pub fn render(&mut self, uniforms: &FrameUniforms, dl: &DrawList) -> Result<(), String> {
        if !self.initialized {
            return Err("E_UI_VULKAN: renderer is not initialized".to_string());
        }
        let scale = uniforms.scale_factor.max(1.0);
        let width = (uniforms.viewport_size[0] * scale).round().max(1.0) as u32;
        let height = (uniforms.viewport_size[1] * scale).round().max(1.0) as u32;
        let needs_rebuild = self.context_lost
            || self
                .swapchain
                .as_ref()
                .map(|s| s.extent.width != width || s.extent.height != height)
                .unwrap_or(true);
        if needs_rebuild {
            unsafe { self.rebuild_swapchain(width, height)? };
            self.context_lost = false;
        }

        unsafe {
            let f = self.device_fns;
            let dev = self.device;
            let mem = self.memory_props;
            let frame = self.frame;

            // Provisional: any failure from here until a successful present must force a
            // swapchain rebuild, otherwise an acquired-but-never-presented image deadlocks
            // the next `vkAcquireNextImageKHR`.
            self.context_lost = true;

            // The previous submission that used these per-frame buffers must finish before the
            // CPU overwrites the persistently mapped memory.
            vk_check(
                (f.vkWaitForFences)(dev, 1, &self.fences[frame], 1, u64::MAX),
                "vkWaitForFences",
            )?;

            let needed = [
                dl.quads.len() * 80,
                dl.ribbons.len() * 48,
                dl.strokes.len() * 48,
                dl.glyphs.len() * 48,
                dl.effects.len() * 80,
                dl.particles.len() * 32,
                dl.clips.len().max(1) * std::mem::size_of::<ClipRect>(),
            ];
            let capacities = [
                self.frames[frame].quads.capacity,
                self.frames[frame].ribbons.capacity,
                self.frames[frame].strokes.capacity,
                self.frames[frame].glyphs.capacity,
                self.frames[frame].effects.capacity,
                self.frames[frame].particles.capacity,
                self.frames[frame].clips.capacity,
            ];
            if needed
                .iter()
                .zip(capacities.iter())
                .any(|(need, cap)| need > cap)
            {
                // Reallocating a buffer that an in-flight frame may still be reading requires idle.
                vk_check((f.vkDeviceWaitIdle)(dev), "vkDeviceWaitIdle(buffer growth)")?;
            }
            let mut changed = false;
            changed |= grow_host_buffer(
                &f,
                dev,
                &mem,
                &mut self.frames[frame].quads,
                dl.quads.len() * 80,
            )?;
            changed |= grow_host_buffer(
                &f,
                dev,
                &mem,
                &mut self.frames[frame].ribbons,
                dl.ribbons.len() * 48,
            )?;
            changed |= grow_host_buffer(
                &f,
                dev,
                &mem,
                &mut self.frames[frame].strokes,
                dl.strokes.len() * 48,
            )?;
            changed |= grow_host_buffer(
                &f,
                dev,
                &mem,
                &mut self.frames[frame].glyphs,
                dl.glyphs.len() * 48,
            )?;
            changed |= grow_host_buffer(
                &f,
                dev,
                &mem,
                &mut self.frames[frame].effects,
                dl.effects.len() * 80,
            )?;
            changed |= grow_host_buffer(
                &f,
                dev,
                &mem,
                &mut self.frames[frame].particles,
                dl.particles.len() * 32,
            )?;
            changed |= grow_host_buffer(
                &f,
                dev,
                &mem,
                &mut self.frames[frame].clips,
                (dl.clips.len().max(1)) * std::mem::size_of::<ClipRect>(),
            )?;
            if changed {
                self.write_descriptors();
            }
            upload_host_buffer(&self.frames[frame].quads, bytes_of(&dl.quads))?;
            upload_host_buffer(&self.frames[frame].ribbons, bytes_of(&dl.ribbons))?;
            upload_host_buffer(&self.frames[frame].strokes, bytes_of(&dl.strokes))?;
            upload_host_buffer(&self.frames[frame].glyphs, bytes_of(&dl.glyphs))?;
            upload_host_buffer(&self.frames[frame].effects, bytes_of(&dl.effects))?;
            upload_host_buffer(&self.frames[frame].particles, bytes_of(&dl.particles))?;
            let mut clip_floats: Vec<f32> = Vec::with_capacity(dl.clips.len().max(1) * 4);
            for c in &dl.clips {
                clip_floats.extend_from_slice(&[c.x, c.y, c.w, c.h]);
            }
            if clip_floats.is_empty() {
                clip_floats.extend_from_slice(&[0.0, 0.0, 1.0, 1.0]);
            }
            upload_host_buffer(&self.frames[frame].clips, bytes_of(&clip_floats))?;

            let mut image_index = 0u32;
            let acquire = (f.vkAcquireNextImageKHR)(
                dev,
                self.swapchain.as_ref().unwrap().handle,
                u64::MAX,
                self.image_available[frame],
                VK_NULL_HANDLE,
                &mut image_index,
            );
            if acquire == VK_ERROR_OUT_OF_DATE_KHR {
                self.context_lost = true;
                return Ok(());
            }
            if acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR {
                return Err(format!(
                    "E_UI_VULKAN: vkAcquireNextImageKHR failed ({acquire})"
                ));
            }
            // SUBOPTIMAL still yields a usable image; the extent check at the top rebuilds the
            // swapchain when the window actually changes size. Do not rebuild every frame on
            // drivers (notably Android) that persistently report SUBOPTIMAL.
            vk_check(
                (f.vkResetCommandBuffer)(self.command_buffers[frame], 0),
                "vkResetCommandBuffer",
            )?;

            let begin = VkCommandBufferBeginInfo {
                sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                pNext: ptr::null(),
                flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                pInheritanceInfo: ptr::null(),
            };
            let cb = self.command_buffers[frame];
            vk_check((f.vkBeginCommandBuffer)(cb, &begin), "vkBeginCommandBuffer")?;
            let sc = self.swapchain.as_ref().unwrap();
            let clear = VkClearValue {
                color: [0.0, 0.0, 0.0, 0.0],
            };
            let pass = VkRenderPassBeginInfo {
                sType: VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                pNext: ptr::null(),
                renderPass: self.render_pass,
                framebuffer: sc.framebuffers[image_index as usize],
                renderArea: VkRect2D {
                    offset: VkOffset2D { x: 0, y: 0 },
                    extent: sc.extent,
                },
                clearValueCount: 1,
                pClearValues: &clear,
            };
            (f.vkCmdBeginRenderPass)(cb, &pass, 0);
            let viewport = VkViewport {
                x: 0.0,
                y: 0.0,
                width: sc.extent.width as f32,
                height: sc.extent.height as f32,
                minDepth: 0.0,
                maxDepth: 1.0,
            };
            let scissor = VkRect2D {
                offset: VkOffset2D { x: 0, y: 0 },
                extent: sc.extent,
            };
            (f.vkCmdSetViewport)(cb, 0, 1, &viewport);
            (f.vkCmdSetScissor)(cb, 0, 1, &scissor);

            let mut push = PushConstants {
                viewport: uniforms.viewport_size,
                atlas: [atlas_width(&self.atlas), atlas_height(&self.atlas)],
                scale: uniforms.scale_factor,
                time: uniforms.time_sec,
                clip_count: dl.clips.len().max(1) as u32,
                base_instance: 0,
            };
            (f.vkCmdPushConstants)(
                cb,
                self.pipeline_layout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                std::mem::size_of::<PushConstants>() as u32,
                &push as *const PushConstants as *const c_void,
            );
            for cmd in &dl.cmds {
                let kind = cmd.kind as usize;
                // Backdrop compositing is not implemented on the Vulkan backend yet; the
                // Metal and GLES backends own true half-res Kawase frosted glass.
                if (kind > 3 && kind != 5 && kind != 6) || cmd.count == 0 {
                    continue;
                }
                let slot = if kind >= 5 { kind - 1 } else { kind };
                let start = cmd.start as usize;
                let max_len = match kind {
                    0 => dl.quads.len(),
                    1 => dl.ribbons.len(),
                    2 => dl.strokes.len(),
                    3 => dl.glyphs.len(),
                    5 => dl.effects.len(),
                    _ => dl.particles.len(),
                };
                if start >= max_len || start + cmd.count as usize > max_len {
                    continue;
                }
                push.base_instance = cmd.start;
                (f.vkCmdPushConstants)(
                    cb,
                    self.pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0,
                    std::mem::size_of::<PushConstants>() as u32,
                    &push as *const PushConstants as *const c_void,
                );
                (f.vkCmdBindPipeline)(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipelines[slot]);
                (f.vkCmdBindDescriptorSets)(
                    cb,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    self.pipeline_layout,
                    0,
                    1,
                    &self.descriptor_sets[frame][slot],
                    0,
                    ptr::null(),
                );
                (f.vkCmdDraw)(cb, 4, cmd.count, 0, 0);
            }
            (f.vkCmdEndRenderPass)(cb);
            vk_check((f.vkEndCommandBuffer)(cb), "vkEndCommandBuffer")?;

            // Reset the fence only once the command buffer is complete, so a recording error
            // cannot leave a permanently unsignaled fence that deadlocks the next frame.
            vk_check(
                (f.vkResetFences)(dev, 1, &self.fences[frame]),
                "vkResetFences",
            )?;
            let wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            let submit = VkSubmitInfo {
                sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                pNext: ptr::null(),
                waitSemaphoreCount: 1,
                pWaitSemaphores: &self.image_available[frame],
                pWaitDstStageMask: &wait_stage,
                commandBufferCount: 1,
                pCommandBuffers: &cb,
                signalSemaphoreCount: 1,
                pSignalSemaphores: &self.present_semaphores[image_index as usize],
            };
            vk_check(
                (f.vkQueueSubmit)(self.queue, 1, &submit, self.fences[frame]),
                "vkQueueSubmit",
            )?;
            let present = VkPresentInfoKHR {
                sType: VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                pNext: ptr::null(),
                waitSemaphoreCount: 1,
                pWaitSemaphores: &self.present_semaphores[image_index as usize],
                swapchainCount: 1,
                pSwapchains: &self.swapchain.as_ref().unwrap().handle,
                pImageIndices: &image_index,
                pResults: ptr::null_mut(),
            };
            let presented = (f.vkQueuePresentKHR)(self.queue, &present);
            if presented == VK_ERROR_OUT_OF_DATE_KHR {
                self.context_lost = true;
            } else if presented != VK_SUCCESS && presented != VK_SUBOPTIMAL_KHR {
                return Err(format!(
                    "E_UI_VULKAN: vkQueuePresentKHR failed ({presented})"
                ));
            } else {
                // A SUBOPTIMAL present has still succeeded; retrying the swapchain forever
                // would burn CPU/GPU on drivers that keep returning it.
                self.context_lost = false;
            }
            self.frame = (self.frame + 1) % FRAMES_IN_FLIGHT;
        }
        Ok(())
    }
}

fn atlas_width(atlas: &Option<Atlas>) -> f32 {
    atlas.as_ref().map(|a| a.width as f32).unwrap_or(1.0)
}
fn atlas_height(atlas: &Option<Atlas>) -> f32 {
    atlas.as_ref().map(|a| a.height as f32).unwrap_or(1.0)
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            if self.device.is_null() {
                return;
            }
            let _ = (self.device_fns.vkDeviceWaitIdle)(self.device);
            self.destroy_swapchain();
            if self.render_pass != VK_NULL_HANDLE {
                (self.device_fns.vkDestroyRenderPass)(self.device, self.render_pass, ptr::null());
            }
            for p in self.pipelines {
                if p != VK_NULL_HANDLE {
                    (self.device_fns.vkDestroyPipeline)(self.device, p, ptr::null());
                }
            }
            if self.pipeline_layout != VK_NULL_HANDLE {
                (self.device_fns.vkDestroyPipelineLayout)(
                    self.device,
                    self.pipeline_layout,
                    ptr::null(),
                );
            }
            if let Some(atlas) = self.atlas.take() {
                (self.device_fns.vkDestroySampler)(self.device, atlas.sampler, ptr::null());
                (self.device_fns.vkDestroyImageView)(self.device, atlas.view, ptr::null());
                (self.device_fns.vkDestroyImage)(self.device, atlas.image, ptr::null());
                (self.device_fns.vkFreeMemory)(self.device, atlas.memory, ptr::null());
            }
            let f = self.device_fns;
            let dev = self.device;
            for frame in &mut self.frames {
                destroy_host_buffer(&f, dev, &mut frame.quads);
                destroy_host_buffer(&f, dev, &mut frame.ribbons);
                destroy_host_buffer(&f, dev, &mut frame.strokes);
                destroy_host_buffer(&f, dev, &mut frame.glyphs);
                destroy_host_buffer(&f, dev, &mut frame.effects);
                destroy_host_buffer(&f, dev, &mut frame.particles);
                destroy_host_buffer(&f, dev, &mut frame.clips);
            }
            if self.descriptor_pool != VK_NULL_HANDLE {
                (self.device_fns.vkDestroyDescriptorPool)(
                    self.device,
                    self.descriptor_pool,
                    ptr::null(),
                );
            }
            if self.descriptor_layout != VK_NULL_HANDLE {
                (self.device_fns.vkDestroyDescriptorSetLayout)(
                    self.device,
                    self.descriptor_layout,
                    ptr::null(),
                );
            }
            for i in 0..FRAMES_IN_FLIGHT {
                if self.fences[i] != VK_NULL_HANDLE {
                    (self.device_fns.vkDestroyFence)(self.device, self.fences[i], ptr::null());
                }
                if self.image_available[i] != VK_NULL_HANDLE {
                    (self.device_fns.vkDestroySemaphore)(
                        self.device,
                        self.image_available[i],
                        ptr::null(),
                    );
                }
            }
            if self.command_pool != VK_NULL_HANDLE {
                (self.device_fns.vkDestroyCommandPool)(self.device, self.command_pool, ptr::null());
            }
            (self.device_fns.vkDestroyDevice)(self.device, ptr::null());
            (self.instance_fns.vkDestroySurfaceKHR)(self.instance, self.surface, ptr::null());
            (self.instance_fns.vkDestroyInstance)(self.instance, ptr::null());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHADERS: [(&str, &[u8], bool); 8] = [
        ("quad.vert.spv", SHADER_QUAD_VERT, true),
        ("quad.frag.spv", SHADER_QUAD_FRAG, false),
        ("ribbon.vert.spv", SHADER_RIBBON_VERT, true),
        ("ribbon.frag.spv", SHADER_RIBBON_FRAG, false),
        ("stroke.vert.spv", SHADER_STROKE_VERT, true),
        ("stroke.frag.spv", SHADER_STROKE_FRAG, false),
        ("glyph.vert.spv", SHADER_GLYPH_VERT, true),
        ("glyph.frag.spv", SHADER_GLYPH_FRAG, false),
    ];

    const SPV_MAGIC: u32 = 0x0723_0203;
    const SPV_OP_NAME: u16 = 5;
    const SPV_OP_MEMBER_NAME: u16 = 6;
    const SPV_OP_ENTRY_POINT: u16 = 15;
    const SPV_OP_DECORATE: u16 = 71;
    const SPV_EXECUTION_MODEL_VERTEX: u32 = 0;
    const SPV_EXECUTION_MODEL_FRAGMENT: u32 = 4;
    const SPV_DECORATION_BINDING: u32 = 33;
    const SPV_DECORATION_DESCRIPTOR_SET: u32 = 34;

    fn words(bytes: &[u8]) -> Vec<u32> {
        assert_eq!(bytes.len() % 4, 0, "SPIR-V must be 32-bit word aligned");
        bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    }

    fn spv_string(inst: &[u32], first_word: usize) -> String {
        let mut out = String::new();
        for word in &inst[first_word..] {
            for byte in word.to_le_bytes() {
                if byte == 0 {
                    return out;
                }
                out.push(byte as char);
            }
        }
        out
    }

    /// Iterate `(opcode, words)` pairs over every instruction in the module.
    fn instructions(module: &[u32]) -> Vec<(u16, &[u32])> {
        assert!(module.len() >= 5);
        let mut out = Vec::new();
        let mut pc = 5;
        while pc < module.len() {
            let header = module[pc] as usize;
            let opcode = (header & 0xFFFF) as u16;
            let word_count = header >> 16;
            assert!(word_count >= 1, "instruction at word {pc} has zero length");
            assert!(pc + word_count <= module.len(), "truncated instruction");
            out.push((opcode, &module[pc..pc + word_count]));
            pc += word_count;
        }
        assert_eq!(pc, module.len());
        out
    }

    #[test]
    fn all_committed_spirv_modules_have_valid_headers_and_stages() {
        for (name, bytes, is_vertex) in SHADERS {
            assert!(bytes.len() > 20, "{name} is too small");
            let module = words(bytes);
            assert_eq!(module[0], SPV_MAGIC, "{name} magic");
            // SPIR-V 1.0/1.1 encoding; glslang currently emits 1.0 for Vulkan 1.1 targets.
            let version = module[1];
            assert!(
                (0x0001_0000..=0x0001_0600).contains(&version),
                "{name} unexpected SPIR-V version {version:#010x}"
            );
            assert_eq!(module[4], 0, "{name} schema must be 0");
            assert!(module[3] > 0, "{name} id bound must be positive");

            let entry_models: Vec<u32> = instructions(&module)
                .into_iter()
                .filter(|(op, _)| *op == SPV_OP_ENTRY_POINT)
                .map(|(_, inst)| inst[1])
                .collect();
            assert_eq!(entry_models.len(), 1, "{name} must have one entry point");
            let expected = if is_vertex {
                SPV_EXECUTION_MODEL_VERTEX
            } else {
                SPV_EXECUTION_MODEL_FRAGMENT
            };
            assert_eq!(entry_models[0], expected, "{name} stage mismatch");
        }
    }

    #[test]
    fn every_module_declares_the_base_instance_push_member() {
        for (name, bytes, _) in SHADERS {
            let module = words(bytes);
            let module_names: Vec<String> = instructions(&module)
                .into_iter()
                .filter(|(op, _)| *op == SPV_OP_NAME)
                .map(|(_, inst)| spv_string(inst, 2))
                .collect();
            assert!(
                module_names.iter().any(|n| n.ends_with("Instance")),
                "{name} does not declare its instance struct"
            );
            let members: Vec<String> = instructions(&module)
                .into_iter()
                .filter(|(op, _)| *op == SPV_OP_MEMBER_NAME)
                .map(|(_, inst)| spv_string(inst, 3))
                .collect();
            assert!(
                members.iter().any(|m| m == "base_instance"),
                "{name} is missing the base_instance push-constant member; \
                 first-instance draws would silently index the wrong instances"
            );
        }
    }

    #[test]
    fn shader_descriptor_bindings_match_the_rust_descriptor_layout() {
        for (name, bytes, _is_vertex) in SHADERS {
            let module = words(bytes);
            let mut sets = Vec::new();
            let mut bindings = Vec::new();
            for (op, inst) in instructions(&module) {
                if op != SPV_OP_DECORATE || inst.len() < 4 {
                    continue;
                }
                match inst[2] {
                    SPV_DECORATION_DESCRIPTOR_SET => sets.push(inst[3]),
                    SPV_DECORATION_BINDING => bindings.push(inst[3]),
                    _ => {}
                }
            }
            assert!(sets.iter().all(|s| *s == 0), "{name} uses a non-zero set");
            assert!(
                bindings.contains(&0),
                "{name} missing instance SSBO (binding 0)"
            );
            assert!(
                bindings.contains(&1),
                "{name} missing clip SSBO (binding 1)"
            );
            if name == "glyph.frag.spv" {
                assert!(
                    bindings.contains(&2),
                    "{name} missing glyph sampler (binding 2)"
                );
            }
        }
    }

    #[test]
    fn push_constant_layout_is_std430_compatible_32_bytes() {
        assert_eq!(std::mem::size_of::<PushConstants>(), 32);
        assert_eq!(std::mem::align_of::<PushConstants>(), 4);
        assert_eq!(std::mem::offset_of!(PushConstants, viewport), 0);
        assert_eq!(std::mem::offset_of!(PushConstants, atlas), 8);
        assert_eq!(std::mem::offset_of!(PushConstants, scale), 16);
        assert_eq!(std::mem::offset_of!(PushConstants, time), 20);
        assert_eq!(std::mem::offset_of!(PushConstants, clip_count), 24);
        assert_eq!(std::mem::offset_of!(PushConstants, base_instance), 28);
    }

    #[test]
    fn instance_and_clip_strides_match_the_shader_structs() {
        // std430: vec4 aligns to 16 B, uint/float to 4 B. These are the exact bytes copied
        // into the SSBOs and consumed as `QuadInstance[]` / `vec4 clips[]`.
        assert_eq!(std::mem::size_of::<sofuu_ui::QuadInstance>(), 80);
        assert_eq!(std::mem::size_of::<sofuu_ui::RibbonInstance>(), 48);
        assert_eq!(std::mem::size_of::<sofuu_ui::StrokeInstance>(), 48);
        assert_eq!(std::mem::size_of::<sofuu_ui::GlyphInstance>(), 48);
        assert_eq!(std::mem::size_of::<sofuu_ui::EffectQuadInstance>(), 80);
        assert_eq!(std::mem::size_of::<sofuu_ui::ParticleInstance>(), 32);
        assert_eq!(std::mem::size_of::<sofuu_ui::BackdropRegion>(), 64);
        assert_eq!(std::mem::size_of::<ClipRect>(), 16);
    }

    #[test]
    fn glsl_450_sources_declare_the_same_bindings() {
        let src = sofuu_ui::shaders::GLSL_450_SRC;
        assert!(src.contains("layout(std430, binding = 0) readonly buffer InstanceBlock"));
        assert!(src.contains("layout(std430, binding = 1) readonly buffer ClipsBlock"));
        assert!(src.contains("layout(binding = 2) uniform sampler2D atlas"));
        assert!(src.contains("layout(push_constant) uniform Push"));
        assert!(src.contains("uint base_instance;"));
        assert!(src.contains("instances[u.base_instance + gl_InstanceIndex]"));
    }
}
