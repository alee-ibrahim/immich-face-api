import sys

if len(sys.argv) > 1 and sys.argv[1] == "admin":
    from .admin import main
elif len(sys.argv) > 1 and sys.argv[1] == "purge":
    from .purge import main as purge_main

    sys.exit(purge_main(sys.argv[2:]))
else:
    from .app import main

main()
