/* ui_c_abi_smoke.c — Phase C2 C integration gate for the embedded UI ABI.
 *
 * Compiled against dist/libsofuu.dylib (default features) by
 * scripts/check_ui_c_abi.sh. It drives the same path a native host does:
 *
 *   session_new -> attach_runtime -> JS mount (sofuu.ui) -> poll_draw_list
 *   -> click -> semantic event -> springs settle (needs_next_frame == 0)
 *   -> session-first free tombstone -> detach -> rt_free
 *
 * Build:  cc scripts/ui_c_abi_smoke.c -Idist -Ldist -lsofuu \
 *            -Wl,-rpath,$PWD/dist -o /tmp/ui_c_abi_smoke
 * Run:    /tmp/ui_c_abi_smoke
 */
#include "sofuu_embed.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);            \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
            failures++;                                                     \
        }                                                                   \
    } while (0)

static void drain_events(SofuuUiSession *ui, int *saw_click) {
    for (;;) {
        char *ev = NULL;
        int rc = sofuu_ui_poll_event_json(ui, &ev);
        if (rc != 1 || ev == NULL) {
            return;
        }
        if (strstr(ev, "\"type\":\"click\"") != NULL &&
            strstr(ev, "\"id\":\"go\"") != NULL) {
            *saw_click = 1;
        }
        sofuu_free(ev);
    }
}

int main(void) {
    CHECK(sofuu_embed_abi_version() >= 2, "unexpected ABI version");

    char *err = NULL;
    SofuuUiSession *ui = sofuu_ui_session_new(NULL, &err);
    if (ui == NULL) {
        fprintf(stderr, "FAIL: sofuu_ui_session_new returned NULL: %s\n",
                err != NULL ? err : "(no error)");
        sofuu_free(err);
        return 1;
    }
    sofuu_free(err);

    CHECK(sofuu_ui_session_set_safe_insets(ui, 24.0f, 8.0f) == SOFUU_OK,
          "set_safe_insets failed");

    const uint8_t *atlas_px = NULL;
    uint32_t atlas_w = 0, atlas_h = 0;
    float atlas_range = 0.0f;
    CHECK(sofuu_ui_session_get_atlas(ui, &atlas_px, &atlas_w, &atlas_h,
                                     &atlas_range) == SOFUU_OK,
          "get_atlas failed");
    CHECK(atlas_px != NULL && atlas_w > 0 && atlas_h > 0 && atlas_range > 0.0f,
          "atlas descriptor is empty");

    /* Mode B is additive and intentionally unresolved until the cross-platform
     * surface renderer lands; every entry must refuse with -2, never crash. */
    CHECK(sofuu_ui_surface_attach_metal(ui, NULL, NULL) == SOFUU_ERR_NOT_IMPL,
          "surface_attach_metal must be -2");
    CHECK(sofuu_ui_surface_attach_egl(ui, NULL, NULL, NULL) == SOFUU_ERR_NOT_IMPL,
          "surface_attach_egl must be -2");
    CHECK(sofuu_ui_surface_render(ui, 1.0f / 60.0f, 800, 600, 1.0f) ==
              SOFUU_ERR_NOT_IMPL,
          "surface_render must be -2");
    CHECK(sofuu_ui_surface_detach(ui) == SOFUU_ERR_NOT_IMPL,
          "surface_detach must be -2");

    SofuuRuntime *rt = sofuu_rt_new(NULL);
    if (rt == NULL) {
        fprintf(stderr, "FAIL: sofuu_rt_new returned NULL\n");
        sofuu_ui_session_free(ui);
        return 1;
    }
    CHECK(sofuu_ui_attach_runtime(rt, ui) == SOFUU_OK, "attach_runtime failed");

    /* A mounted view starts src/js/ui.js's 16 ms event pump. The blocking
     * eval path must therefore NOT wait for libuv to go idle; this call
     * returning proves the attached-session branch works. */
    char *out = NULL;
    const char *mount =
        "sofuu.ui.app('C ABI').add(sofuu.ui.button('go', 'Go')).mount();"
        "sofuu.ui.isAttached()";
    CHECK(sofuu_rt_eval(rt, mount, &out) == SOFUU_OK, "mount eval failed");
    CHECK(out != NULL && strstr(out, "true") != NULL,
          "sofuu.ui.isAttached() was not true after attach");
    sofuu_free(out);

    SofuuUiDrawList dl;
    memset(&dl, 0, sizeof dl);
    CHECK(sofuu_ui_poll_draw_list(ui, 1.0f / 60.0f, 800.0f, 600.0f, 2.0f,
                                  &dl) == SOFUU_OK,
          "poll_draw_list failed");
    CHECK(dl.quad_count > 0 && dl.cmd_count > 0 && dl.clip_count > 0,
          "mounted scene produced an empty DrawList");
    CHECK(dl.uniforms.viewport_size[0] == 800.0f &&
              dl.uniforms.viewport_size[1] == 600.0f,
          "viewport uniform mismatch");
    CHECK(dl.quads != NULL && dl.cmds != NULL && dl.clips != NULL,
          "DrawList borrowed slices are NULL");

    /* The root column stretches the button across the viewport; scan a few
     * rows so this stays independent of the theme's exact control height. */
    int saw_click = 0;
    for (float y = 8.0f; y <= 160.0f && !saw_click; y += 4.0f) {
        SofuuUiInputEvent down;
        memset(&down, 0, sizeof down);
        down.kind = SOFUU_INPUT_POINTER_DOWN;
        down.x = 400.0f;
        down.y = y;
        down.button = 0;
        down.text_utf8 = NULL;
        SofuuUiInputEvent up = down;
        up.kind = SOFUU_INPUT_POINTER_UP;

        CHECK(sofuu_ui_dispatch_input(ui, &down) == SOFUU_OK,
              "pointer down rejected");
        CHECK(sofuu_ui_dispatch_input(ui, &up) == SOFUU_OK,
              "pointer up rejected");
        CHECK(sofuu_ui_poll_draw_list(ui, 1.0f / 60.0f, 800.0f, 600.0f, 2.0f,
                                      &dl) == SOFUU_OK,
              "poll after click failed");
        drain_events(ui, &saw_click);
    }
    CHECK(saw_click, "pointer down/up never produced a click event for 'go'");

    int settled = 0;
    for (int i = 0; i < 600; i++) {
        CHECK(sofuu_ui_poll_draw_list(ui, 1.0f / 60.0f, 800.0f, 600.0f, 2.0f,
                                      &dl) == SOFUU_OK,
              "settle poll failed");
        drain_events(ui, &saw_click);
        if (dl.needs_next_frame == 0) {
            settled = 1;
            break;
        }
    }
    CHECK(settled, "springs never settled to needs_next_frame == 0");

    /* Zero-boilerplate Auto-Composer through the same JS attach: the five
     * __sofuu_ui_auto_* funnels must work without libsofuu_gpu. */
    out = NULL;
    CHECK(sofuu_rt_eval(
              rt,
              "sofuu.ui.autoComposer({title:'Auto C ABI', theme:'dark'}).mount();"
              "JSON.parse(__sofuu_ui_auto_scene()).title",
              &out) == SOFUU_OK,
          "autoComposer mount failed");
    CHECK(out != NULL && strstr(out, "Auto C ABI") != NULL,
          "auto scene did not pick up its title");
    sofuu_free(out);
    CHECK(sofuu_ui_poll_draw_list(ui, 1.0f / 60.0f, 800.0f, 600.0f, 2.0f,
                                  &dl) == SOFUU_OK,
          "poll after auto mount failed");
    CHECK(dl.quad_count > 0 && dl.cmd_count > 0,
          "Auto-Composer scene produced an empty DrawList");

    /* Session-first free: the JS globals must tombstone (false) instead of
     * touching freed memory. Use the non-blocking eval: the mounted view's
     * pump is no longer attached to a live session, so an until-idle drain
     * would wait on the interval forever. */
    sofuu_ui_session_free(ui);
    ui = NULL;
    out = NULL;
    CHECK(sofuu_rt_eval_nb(rt, "sofuu.ui.isAttached()", &out) == SOFUU_OK,
          "post-free eval failed");
    CHECK(out != NULL && strstr(out, "false") != NULL,
          "freed session did not report detached");
    sofuu_free(out);

    CHECK(sofuu_ui_detach_runtime(rt) == SOFUU_OK, "detach_runtime failed");
    sofuu_rt_free(rt);

    if (failures != 0) {
        fprintf(stderr, "ui_c_abi_smoke: %d failure(s)\n", failures);
        return 1;
    }
    printf("ui_c_abi_smoke: OK (mount, %u quads, click, settled, tombstone)\n",
           dl.quad_count);
    return 0;
}
