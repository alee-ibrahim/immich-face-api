import sys

if len(sys.argv) > 1 and sys.argv[1] == "admin":
    from .admin import main
else:
    from .app import main

main()
