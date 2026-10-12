/* Development-only stub of the libsofuu C ABI (v2).
 *
 * Exists purely so `swiftc -typecheck` can validate the app's Swift without
 * a built dist/libsofuu.xcframework. The declarations below are copied
 * verbatim (same names, same types, same order) from dist/sofuu_embed.h —
 * only the unused surface is omitted. Never shipped, never linked.
 *
 *   swiftc -typecheck -parse-as-library \
 *     -sdk "$(xcrun --sdk iphoneos --show-sdk-path)" \
 *     -target arm64-apple-ios16.0 -I .devstub \
 *     $(find SofuuApp -name '*.swift')
 */
#ifndef SOFUU_EMBED_STUB_H
#define SOFUU_EMBED_STUB_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SofuuRuntime SofuuRuntime;
typedef void (*sofuu_event_cb)(const char *event_json, void *opaque);

uint32_t sofuu_embed_abi_version(void);

SofuuRuntime *sofuu_rt_new(const char *config_json);
void sofuu_rt_free(SofuuRuntime *rt);

int sofuu_rt_call(SofuuRuntime *rt, const char *method,
                  const char *args_json, char **out_json);

int sofuu_rt_call_stream(SofuuRuntime *rt, const char *method,
                         const char *args_json,
                         sofuu_event_cb on_event, void *opaque,
                         uint64_t *out_cancel_id);

int sofuu_rt_cancel(SofuuRuntime *rt, uint64_t cancel_id);

int sofuu_rt_eval(SofuuRuntime *rt, const char *source, char **out_json);

int sofuu_embed_info(SofuuRuntime *rt, char **out_json);

void sofuu_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* SOFUU_EMBED_STUB_H */
