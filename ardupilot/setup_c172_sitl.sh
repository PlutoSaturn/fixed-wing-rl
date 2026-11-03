#!/usr/bin/env bash
# setup_c172_sitl.sh - Install JSBSim's Cessna 172 (c172x) as an ArduPilot SITL aircraft.
#
# Run once in WSL (Ubuntu), after building JSBSim from source, from the project folder:
#     cd /mnt/c/Users/isaia/fixed-wing-rl
#     bash ardupilot/setup_c172_sitl.sh [path-to-jsbsim-source] [path-to-ardupilot]
# defaults: ~/jsbsim and ~/ardupilot
#
# What it does
#   1. checks the JSBSim program is on your PATH (ArduPilot launches it by name)
#   2. copies c172x into ardupilot/Tools/autotest/aircraft/c172x/, with the engine,
#      propeller and system files it uses (ArduPilot runs JSBSim from Tools/autotest,
#      so the model must be found there)
#   3. adds a small system file that starts the engine and keeps it running:
#      ArduPilot only sets "engine running" without turning the engine over, so
#      the c172x piston engine would stop at once. The file keeps both magnetos on
#      and runs the starter until the engine is running (in about 2 s)
#   4. test-loads the model with JSBSim from ArduPilot's folder
#
# ArduPilot writes the start-position file (aircraft/c172x/reset.xml) itself each
# time SITL starts, so it is not created here.
#
# It is safe to run again; it overwrites its own previous install.
set -euo pipefail

JSBSIM_SRC="${1:-$HOME/jsbsim}"
ARDUPILOT="${2:-$HOME/ardupilot}"
AUTOTEST="$ARDUPILOT/Tools/autotest"
DEST="$AUTOTEST/aircraft/c172x"

echo "JSBSim source : $JSBSIM_SRC"
echo "ArduPilot     : $ARDUPILOT"

# ---- 1. checks --------------------------------------------------------------------
if ! command -v JSBSim >/dev/null 2>&1; then
    echo "ERROR: the JSBSim program is not on your PATH."
    echo "Build it and add it to PATH:  echo 'export PATH=$HOME/jsbsim/build/src:$PATH' >> ~/.bashrc; source ~/.bashrc"
    exit 1
fi
if JSBSim --version 2>&1 | grep -qi "python"; then
    echo "WARNING: 'JSBSim' on your PATH looks like the pip/Python version. ArduPilot passes"
    echo "         options it does not accept. Use the C++ build: ~/jsbsim/build/src/JSBSim"
fi
for f in "$JSBSIM_SRC/aircraft/c172x/c172x.xml" "$JSBSIM_SRC/engine/eng_io320.xml" \
         "$JSBSIM_SRC/engine/prop_75in2f.xml" "$JSBSIM_SRC/systems/GNCUtilities.xml" \
         "$JSBSIM_SRC/systems/Autopilot.xml"; do
    [ -f "$f" ] || { echo "ERROR: missing $f (is $JSBSIM_SRC the JSBSim source folder?)"; exit 1; }
done
[ -d "$AUTOTEST/aircraft" ] || { echo "ERROR: $AUTOTEST/aircraft not found"; exit 1; }

# ---- 2. copy the aircraft and the files it references -------------------------------
rm -rf "$DEST"
cp -r "$JSBSIM_SRC/aircraft/c172x" "$DEST"
mkdir -p "$DEST/Engines" "$DEST/Systems" "$AUTOTEST/engine" "$AUTOTEST/systems"
for f in eng_io320.xml prop_75in2f.xml; do
    cp "$JSBSIM_SRC/engine/$f" "$DEST/Engines/"
    cp "$JSBSIM_SRC/engine/$f" "$AUTOTEST/engine/"         # JSBSim's global engine folder (fallback)
done
for f in GNCUtilities.xml Autopilot.xml; do
    cp "$JSBSIM_SRC/systems/$f" "$DEST/Systems/"
    cp "$JSBSIM_SRC/systems/$f" "$AUTOTEST/systems/"       # JSBSim's global systems folder (fallback)
done

# ---- 3. keep the engine running (magnetos on, starter until it runs) -------------------
cat > "$DEST/Systems/ardupilot_engine.xml" <<'XML'
<?xml version="1.0"?>
<!-- Added for ArduPilot SITL. ArduPilot only sets propulsion/engine[0]/set-running = 1,
     which leaves the engine at 0 rpm, so the c172x piston engine stops again at once.
     This keeps both magnetos on and runs the starter whenever the engine is not
     running, so it starts by itself (in about 2 s) and restarts if it ever stalls.
     Mixture is already set automatically by c172x's own "Mixture control" system. -->
<system name="ArduPilot engine settings">
  <channel name="Ignition">
    <switch name="systems/ardupilot/magnetos">
      <default value="3"/>
      <output>propulsion/magneto_cmd</output>
    </switch>
    <switch name="systems/ardupilot/starter">
      <default value="0"/>
      <test value="1">
        propulsion/engine[0]/set-running eq 0
      </test>
      <output>propulsion/starter_cmd</output>
    </switch>
  </channel>
</system>
XML
cp "$DEST/Systems/ardupilot_engine.xml" "$AUTOTEST/systems/"
if ! grep -q 'system file="ardupilot_engine"' "$DEST/c172x.xml"; then
    sed -i 's#<system file="GNCUtilities"/>#<system file="GNCUtilities"/>\n    <system file="ardupilot_engine"/>#' "$DEST/c172x.xml"
fi
grep -q 'system file="ardupilot_engine"' "$DEST/c172x.xml" || { echo "ERROR: could not add the engine system to c172x.xml"; exit 1; }

# ---- 4. test-load the model from ArduPilot's folder -------------------------------------
echo
echo "Test: loading c172x with JSBSim from $AUTOTEST ..."
if (cd "$AUTOTEST" && JSBSim --aircraft=c172x --initfile=reset00 --end=2 > /tmp/c172x_test.log 2>&1); then
    if grep -qiE "could not|unable to|not found|fatal|^\s*error" /tmp/c172x_test.log; then
        echo "WARNING: JSBSim reported problems; see /tmp/c172x_test.log"
    else
        echo "OK: c172x loads and runs."
    fi
else
    echo "ERROR: JSBSim failed to run c172x; see /tmp/c172x_test.log"
    exit 1
fi
echo
echo "Installed: $DEST"
echo "Next: generate the test mission and fly it (docs/ArduPilot_Commands.md)."
