"""Docker entrypoint: fix /app/data ownership, then drop from root to the
unprivileged runtime user before running the actual command.

/app/data is chowned to the trebuchet user at image build time, but that
ownership only applies to the image's own layer - a bind mount (a host
folder via DATA_DIR, or an Unraid appdata share) or a fresh named volume
brings its own ownership instead, which almost never matches the uid
inside the container. The non-root user then can't write user_defaults.json
and the app crashes with PermissionError. Running this once as root at
container start, then exec'ing the real command as the unprivileged user,
fixes that regardless of what the mount's ownership happened to be.
"""

import os
import pwd
import sys

DATA_DIR = "/app/data"
RUN_AS_USER = "trebuchet"


def main() -> None:
    user = pwd.getpwnam(RUN_AS_USER)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.chown(DATA_DIR, user.pw_uid, user.pw_gid)
    for name in os.listdir(DATA_DIR):
        os.chown(os.path.join(DATA_DIR, name), user.pw_uid, user.pw_gid)

    os.setgid(user.pw_gid)
    os.setuid(user.pw_uid)
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
