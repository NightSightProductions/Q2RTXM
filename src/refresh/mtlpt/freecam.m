/*
Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
Copyright (C) 2024 Quake II RTX Metal port contributors

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

// Free camera for the paused game (photo mode), the port of vkpt/freecam.c.
// All game input is redirected here while the game is paused; WASDQE move the
// camera, mouse buttons rotate/roll/zoom, and the wheel drives depth of field.

#include "mtlpt_metal.h"
#include "../../client/client.h"

static vec3_t freecam_vieworg;
static vec3_t freecam_viewangles;
static float  freecam_zoom = 1.0f;
static bool   freecam_keystate[6];
static bool   freecam_active;
static int    freecam_player_model;

extern float autosens_x;
extern float autosens_y;
extern cvar_t *m_accel;
extern cvar_t *m_autosens;
extern cvar_t *m_pitch;
extern cvar_t *m_invert;
extern cvar_t *m_yaw;
extern cvar_t *sensitivity;

cvar_t *cvar_pt_dof;
cvar_t *cvar_pt_aperture;
cvar_t *cvar_pt_focus;
cvar_t *cvar_pt_freecam;

void mtl_freecam_init(void)
{
    cvar_pt_dof = Cvar_Get("pt_dof", "1", CVAR_ARCHIVE);
    cvar_pt_aperture = Cvar_Get("pt_aperture", "2.0", 0);
    cvar_pt_focus = Cvar_Get("pt_focus", "200", 0);
    cvar_pt_freecam = Cvar_Get("pt_freecam", "1", CVAR_ARCHIVE);
}

void mtl_freecam_reset(void)
{
    if (!freecam_active)
        return;

    Cvar_SetByVar(cl_player_model, va("%d", freecam_player_model), FROM_CODE);
    freecam_active = false;
}

static void freecam_mousemove(void)
{
    int dx, dy;
    float mx, my, speed;

    if (!vid.get_mouse_motion || !vid.get_mouse_motion(&dx, &dy))
        return;

    mx = (float)dx;
    my = (float)dy;
    if (!mx && !my)
        return;

    if (Key_IsDown(K_MOUSE1) && Key_IsDown(K_MOUSE2)) {
        mx *= sensitivity->value;
        freecam_viewangles[ROLL] += m_yaw->value * mx;
    } else if (Key_IsDown(K_MOUSE1)) {
        Cvar_ClampValue(m_accel, 0, 1);

        speed = sqrtf(mx * mx + my * my);
        speed = sensitivity->value + speed * m_accel->value;
        mx *= speed;
        my *= speed;

        if (m_autosens->integer) {
            mx *= cl.fov_x * autosens_x;
            my *= cl.fov_y * autosens_y;
        }

        mx /= freecam_zoom;
        my /= freecam_zoom;

        freecam_viewangles[YAW] -= m_yaw->value * mx;
        freecam_viewangles[PITCH] += m_pitch->value * my * (m_invert->integer ? -1.0f : 1.0f);
        freecam_viewangles[PITCH] = max(-90.0f, min(90.0f, freecam_viewangles[PITCH]));
    } else if (Key_IsDown(K_MOUSE2)) {
        freecam_zoom *= powf(0.5f, my * m_pitch->value * 0.1f);
        freecam_zoom = max(0.5f, min(20.0f, freecam_zoom));
    }
}

// Returns true when the camera changed and temporal history must be dropped.
bool mtl_freecam_update(refdef_t *fd, float frame_time)
{
    if (cl_paused->integer != 2 || !sv_paused->integer || !cvar_pt_freecam->integer) {
        mtl_freecam_reset();
        return false;
    }

    if (!freecam_active) {
        VectorCopy(fd->vieworg, freecam_vieworg);
        VectorCopy(fd->viewangles, freecam_viewangles);
        freecam_zoom = 1.0f;
        freecam_player_model = cl_player_model->integer;
        freecam_active = true;
    }

    vec3_t prev_vieworg, prev_viewangles;
    VectorCopy(freecam_vieworg, prev_vieworg);
    VectorCopy(freecam_viewangles, prev_viewangles);
    float prev_zoom = freecam_zoom;

    vec3_t velocity = { 0.0f, 0.0f, 0.0f };
    if (freecam_keystate[0]) velocity[0] += 1.0f;
    if (freecam_keystate[1]) velocity[0] -= 1.0f;
    if (freecam_keystate[2]) velocity[1] += 1.0f;
    if (freecam_keystate[3]) velocity[1] -= 1.0f;
    if (freecam_keystate[4]) velocity[2] += 1.0f;
    if (freecam_keystate[5]) velocity[2] -= 1.0f;

    if (Key_IsDown(K_SHIFT))
        VectorScale(velocity, 5.0f, velocity);
    else if (Key_IsDown(K_CTRL))
        VectorScale(velocity, 0.1f, velocity);

    vec3_t forward, right, up;
    AngleVectors(freecam_viewangles, forward, right, up);
    const float speed = 100.0f;
    VectorMA(freecam_vieworg, velocity[0] * frame_time * speed, forward, freecam_vieworg);
    VectorMA(freecam_vieworg, velocity[1] * frame_time * speed, right, freecam_vieworg);
    VectorMA(freecam_vieworg, velocity[2] * frame_time * speed, up, freecam_vieworg);

    freecam_mousemove();

    VectorCopy(freecam_vieworg, fd->vieworg);
    VectorCopy(freecam_viewangles, fd->viewangles);
    fd->fov_x = RAD2DEG(atanf(tanf(DEG2RAD(fd->fov_x) * 0.5f) / freecam_zoom)) * 2.0f;
    fd->fov_y = RAD2DEG(atanf(tanf(DEG2RAD(fd->fov_y) * 0.5f) / freecam_zoom)) * 2.0f;

    bool moved = false;
    if (!VectorCompare(freecam_vieworg, prev_vieworg) || !VectorCompare(freecam_viewangles, prev_viewangles)) {
        if (freecam_player_model != CL_PLAYER_MODEL_DISABLED && cl_player_model->integer != CL_PLAYER_MODEL_THIRD_PERSON)
            Cvar_SetByVar(cl_player_model, va("%d", CL_PLAYER_MODEL_THIRD_PERSON), FROM_CODE);
        moved = true;
    }
    if (freecam_zoom != prev_zoom)
        moved = true;

    return moved;
}

bool R_InterceptKey_MTL(unsigned key, bool down)
{
    if (cl_paused->integer != 2 || !sv_paused->integer)
        return false;

    const char *kb = Key_GetBindingForKey(key);
    if (kb && strstr(kb, "pause"))
        return false;

    if (cvar_pt_dof->integer != 0 && down && (key == K_MWHEELUP || key == K_MWHEELDOWN)) {
        cvar_t *var;
        float minvalue, maxvalue;

        if (Key_IsDown(K_SHIFT)) {
            var = cvar_pt_aperture;
            minvalue = 0.01f;
            maxvalue = 10.0f;
        } else {
            var = cvar_pt_focus;
            minvalue = 1.0f;
            maxvalue = 10000.0f;
        }

        float factor = Key_IsDown(K_CTRL) ? 1.01f : 1.1f;
        if (key == K_MWHEELDOWN)
            factor = 1.0f / factor;

        float value = Q_clipf(var->value * factor, minvalue, maxvalue);
        Cvar_SetByVar(var, va("%f", value), FROM_CONSOLE);
        return true;
    }

    switch (key) {
    case 'w': freecam_keystate[0] = down; return true;
    case 's': freecam_keystate[1] = down; return true;
    case 'd': freecam_keystate[2] = down; return true;
    case 'a': freecam_keystate[3] = down; return true;
    case 'e': freecam_keystate[4] = down; return true;
    case 'q': freecam_keystate[5] = down; return true;

    // Keys the freecam uses must not reach the game (MOUSE1 is usually fire).
    case K_CTRL:
    case K_SHIFT:
    case K_MWHEELDOWN:
    case K_MWHEELUP:
    case K_MOUSE1:
    case K_MOUSE2:
        return true;
    }

    return false;
}
