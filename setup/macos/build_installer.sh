#!/bin/bash
#
# Builds the macOS distribution of Quake II RTX, the counterpart of the
# Windows setup.nsi installer:
#
#   Quake2RTX-macOS.dmg
#     Install Quake II RTX.app   installer wizard (Installer.swift) carrying
#       Quake II RTX.app         the game: engine, Metal shaders, media
#       shareware/               the Quake II shareware demo files
#
# Requirements: Xcode with the Metal toolchain
#   (xcodebuild -downloadComponent MetalToolchain), CMake and Ninja.
#
# Inputs that are not in git, as for the Windows installer:
#   baseq2/q2rtx_media.pkz, baseq2/blue_noise.pkz   from a Q2RTX release
#   baseq2/shareware/pak0.pak + players/            the shareware demo files
#
# Usage: setup/macos/build_installer.sh [output dir]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SETUP="${ROOT}/setup/macos"
OUT="${1:-${ROOT}/build-macos-dist}"
BUILD="${OUT}/build"
STAGE="${OUT}/stage"
MIN_MACOS="13.0"    # Metal Shading Language 3.0, used by the renderer

APP_NAME="Quake II RTX"
APP="${STAGE}/payload/${APP_NAME}.app"
INSTALLER="${STAGE}/dmg/Install ${APP_NAME}.app"
DMG="${OUT}/Quake2RTX-macOS.dmg"

die() { echo "error: $*" >&2; exit 1; }

#
# Inputs
#
for f in q2rtx_media.pkz blue_noise.pkz; do
    [[ -f "${ROOT}/baseq2/${f}" ]] || die "baseq2/${f} is missing (it comes with a Q2RTX release)"
done

# The same source as setup.nsi and the Linux CPack "shareware" component.
SHAREWARE="${ROOT}/baseq2/shareware"
[[ -f "${SHAREWARE}/pak0.pak" && -d "${SHAREWARE}/players" ]] ||
    die "the shareware demo files are missing: put the Quake II demo's pak0.pak and players/ in baseq2/shareware/"
# Only the freely distributable demo may be bundled, never the retail game.
if LC_ALL=C grep -q "maps/base1.bsp" "${SHAREWARE}/pak0.pak"; then
    die "${SHAREWARE}/pak0.pak is from the full game and must not be distributed; provide the shareware demo pak"
fi

xcrun -sdk macosx metal --version >/dev/null 2>&1 ||
    die "the Metal compiler is not available (install Xcode and run: xcodebuild -downloadComponent MetalToolchain)"

#
# Build: engine, game library and the Metal shader library
#
cmake -S "${ROOT}" -B "${BUILD}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${MIN_MACOS}" \
    -DCONFIG_MTLPT_RENDERER=ON -DCONFIG_VKPT_RENDERER=OFF -DCONFIG_GL_RENDERER=ON \
    -DCONFIG_USE_CURL=OFF
cmake --build "${BUILD}" --target client game metal_shaders

VERSION="$(git -C "${ROOT}" describe --tags --always 2>/dev/null || echo 1.0)"
VERSION="${VERSION#v}"

#
# Quake II RTX.app
#
rm -rf "${STAGE}"
mkdir -p "${APP}/Contents/MacOS" "${APP}/Contents/Resources/baseq2"
RES="${APP}/Contents/Resources"

sed -e "s/@VERSION@/${VERSION}/g" -e "s/@MIN_MACOS@/${MIN_MACOS}/g" "${SETUP}/Info.plist" > "${APP}/Contents/Info.plist"
cp "${ROOT}/q2rtx" "${APP}/Contents/MacOS/q2rtx"

# Game data, the files setup.nsi installs: the game library, the media and
# noise packages and the shaders, plus the loose files tracked in git
# (menus, configs, materials) so they match this build.
cp "${ROOT}/baseq2/gameaarch64.dylib" "${ROOT}/baseq2/q2rtx_media.pkz" "${ROOT}/baseq2/blue_noise.pkz" "${RES}/baseq2/"
mkdir -p "${RES}/baseq2/shader_mtlpt"
cp "${ROOT}/baseq2/shader_mtlpt/q2rtx.metallib" "${RES}/baseq2/shader_mtlpt/"
(cd "${ROOT}" && git ls-files baseq2 rogue | while read -r f; do
    mkdir -p "${RES}/$(dirname "$f")" && cp "$f" "${RES}/$f"
done)
[[ -f "${ROOT}/rogue/q2rtx_media.pkz" ]] && cp "${ROOT}/rogue/q2rtx_media.pkz" "${RES}/rogue/"
cp "${ROOT}/license.txt" "${ROOT}/notice.txt" "${ROOT}/readme.md" "${ROOT}/changelog.md" "${RES}/"

# Icon, from the 256x256 Linux icon
ICONSET="${STAGE}/q2rtx.iconset"
mkdir -p "${ICONSET}"
for s in 16 32 128 256 512; do
    sips -z "$s" "$s" "${ROOT}/setup/q2rtx.png" --out "${ICONSET}/icon_${s}x${s}.png" >/dev/null
    d=$((s * 2))
    sips -z "$d" "$d" "${ROOT}/setup/q2rtx.png" --out "${ICONSET}/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "${ICONSET}" -o "${RES}/q2rtx.icns"

# Ad hoc signature: Apple Silicon only runs signed code. Distribution outside
# this Mac needs a Developer ID signature and notarization instead, otherwise
# Gatekeeper asks users to open the installer with right click > Open.
SIGN_ID="${SIGN_ID:--}"
SIGN_OPTS=()
# Notarization needs the hardened runtime; with an ad hoc signature its
# library validation would refuse to load the game library.
[[ "${SIGN_ID}" != "-" ]] && SIGN_OPTS=(--options runtime --timestamp)
codesign --force --sign "${SIGN_ID}" "${SIGN_OPTS[@]+"${SIGN_OPTS[@]}"}" "${RES}/baseq2/gameaarch64.dylib"
codesign --force --sign "${SIGN_ID}" "${SIGN_OPTS[@]+"${SIGN_OPTS[@]}"}" "${APP}"

#
# Install Quake II RTX.app
#
# A native wizard with the pages of setup.nsi (Installer.swift).
mkdir -p "${INSTALLER}/Contents/MacOS" "${INSTALLER}/Contents/Resources"
sed -e "s/@VERSION@/${VERSION}/g" -e "s/@MIN_MACOS@/${MIN_MACOS}/g" "${SETUP}/InstallerInfo.plist" > "${INSTALLER}/Contents/Info.plist"
xcrun swiftc -O -swift-version 5 -target "arm64-apple-macos${MIN_MACOS}" "${SETUP}/Installer.swift" -o "${INSTALLER}/Contents/MacOS/installer"
cp "${RES}/q2rtx.icns" "${INSTALLER}/Contents/Resources/q2rtx.icns"
cp "${ROOT}/setup/WelcomeImage.bmp" "${INSTALLER}/Contents/Resources/WelcomeImage.bmp"
ditto "${APP}" "${INSTALLER}/Contents/Resources/${APP_NAME}.app"
mkdir -p "${INSTALLER}/Contents/Resources/shareware"
cp "${SHAREWARE}/pak0.pak" "${INSTALLER}/Contents/Resources/shareware/"
ditto "${SHAREWARE}/players" "${INSTALLER}/Contents/Resources/shareware/players"
codesign --force --sign "${SIGN_ID}" "${SIGN_OPTS[@]+"${SIGN_OPTS[@]}"}" "${INSTALLER}"

#
# Disk image
#
cp "${ROOT}/license.txt" "${STAGE}/dmg/License.txt"
rm -f "${DMG}"
hdiutil create -volname "${APP_NAME}" -srcfolder "${STAGE}/dmg" -ov -format UDZO "${DMG}" >/dev/null
echo "Created ${DMG}"
