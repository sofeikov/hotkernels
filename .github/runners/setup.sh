#!/bin/bash
# Installs GitHub Actions runners for this repository on a Mac with Apple
# Silicon, or removes them. The macOS jobs of ci.yml and release.yml run on
# them when one is online (see .github/workflows/pick-runner.yml).
#
#   .github/runners/setup.sh [COUNT]   install COUNT runners (default 4)
#   .github/runners/setup.sh remove    remove every runner this script installed
#
# A runner takes one job at a time, so COUNT is how many jobs run at once.
# Running it again is safe: a runner already installed is left as it is, and
# started if it is not running, so a larger COUNT adds runners. To reinstall
# one, remove them all first.
#
# Before the first run, create the account the runners run as: a standard,
# non-admin user, so that a job cannot change the machine.
#
#   sudo sysadminctl -addUser ci-runner -fullName "CI Runner" -password -
#
# Run this script as yourself, not with sudo. It uses your `gh` login to get
# registration tokens, which needs admin rights on the repository, and asks
# for your password when it needs sudo.
#
# Each runner is a launchd system service: it starts at boot, without anyone
# logged in as ci-runner, and is restarted if it exits. With FileVault on,
# "at boot" means once someone has unlocked the disk at the login screen.
set -euo pipefail

REPO="sofeikov/hotkernels"
# The account the runners run as.
RUNNER_USER="ci-runner"
# The label pick-runner.yml looks for.
LABEL="hotkernels-mac"
# Prefix of the launchd service names and of the runners' directories.
SERVICE_PREFIX="com.hotkernels.actions-runner"
DIR_PREFIX="/Users/${RUNNER_USER}/actions-runner"
DEFAULT_COUNT=4

# The first runner has no suffix; the others are numbered from 2.
suffix() {
    if [ "$1" -eq 1 ]; then echo ""; else echo "-$1"; fi
}

install_runner() {
    local number="$1" version="$2"
    local tail dir service plist archive url token
    tail="$(suffix "${number}")"
    dir="${DIR_PREFIX}${tail}"
    service="${SERVICE_PREFIX}${tail}"
    plist="/Library/LaunchDaemons/${service}.plist"
    archive="actions-runner-osx-arm64-${version}.tar.gz"
    url="https://github.com/actions/runner/releases/download/v${version}/${archive}"

    echo "Runner ${number}: ${dir}"
    # A runner that is configured and running is not touched: stopping it would
    # kill the job it is running.
    if sudo test -e "${dir}/.runner" && sudo launchctl print "system/${service}" >/dev/null 2>&1; then
        echo "  already installed and running"
        return
    fi
    # config.sh writes .runner when it configures a directory, and refuses to
    # configure that directory again, whatever --replace says (that flag only
    # replaces a registration of the same name on GitHub). A configured runner
    # that is not running only needs its service.
    if ! sudo test -e "${dir}/.runner"; then
        # A service left loaded on an unconfigured directory runs nothing
        # useful; stop it so the bootstrap below can load it afresh.
        sudo launchctl bootout "system/${service}" 2>/dev/null || true
        # Valid for one hour.
        token="$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq .token)"
        sudo -u "${RUNNER_USER}" -H bash -euo pipefail <<INNER
mkdir -p "${dir}"
cd "${dir}"
if [ ! -x ./config.sh ]; then
    curl -fsSL -o "${archive}" "${url}"
    tar xzf "${archive}"
    rm "${archive}"
fi
./config.sh --unattended --replace \
    --url "https://github.com/${REPO}" --token "${token}" \
    --name "\$(hostname -s)${tail}" --labels "${LABEL}"
INNER
    fi

    sudo tee "${plist}" >/dev/null <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>${service}</string>
    <key>UserName</key><string>${RUNNER_USER}</string>
    <key>WorkingDirectory</key><string>${dir}</string>
    <key>ProgramArguments</key>
    <array><string>${dir}/run.sh</string></array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key><string>/Users/${RUNNER_USER}</string>
        <key>PATH</key><string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>${dir}/runner.out.log</string>
    <key>StandardErrorPath</key><string>${dir}/runner.err.log</string>
</dict>
</plist>
PLIST
    sudo chown root:wheel "${plist}"
    sudo chmod 644 "${plist}"
    sudo launchctl bootstrap system "${plist}"
}

remove_runners() {
    local plist service tail dir token
    for plist in /Library/LaunchDaemons/"${SERVICE_PREFIX}"*.plist; do
        [ -e "${plist}" ] || continue
        service="$(basename "${plist}" .plist)"
        tail="${service#"${SERVICE_PREFIX}"}"
        dir="${DIR_PREFIX}${tail}"
        echo "Removing ${service}"
        sudo launchctl bootout "system/${service}" 2>/dev/null || true
        sudo rm "${plist}"
        token="$(gh api -X POST "repos/${REPO}/actions/runners/remove-token" --jq .token)"
        sudo -u "${RUNNER_USER}" -H bash -c "cd '${dir}' && ./config.sh remove --token '${token}'" || true
        sudo rm -rf "${dir}"
    done
}

if ! id "${RUNNER_USER}" >/dev/null 2>&1; then
    echo "The account ${RUNNER_USER} does not exist. Create it first; see the top of this script." >&2
    exit 1
fi

if [ "${1:-}" = "remove" ]; then
    remove_runners
    exit 0
fi

COUNT="${1:-${DEFAULT_COUNT}}"
VERSION="$(gh api repos/actions/runner/releases/latest --jq .tag_name | sed 's/^v//')"
for NUMBER in $(seq 1 "${COUNT}"); do
    install_runner "${NUMBER}" "${VERSION}"
done
echo "Done. The runners should show as online within a minute:"
echo "  gh api repos/${REPO}/actions/runners --jq '.runners[] | [.name, .status] | @tsv'"
