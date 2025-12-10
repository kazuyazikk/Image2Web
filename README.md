Image2Web is a web application that can turn physical hand-drawn sketch wireframes into actual HTML and CSS code! Utilizes Python, TensorFlow, and Keras for model training along with the code generation. Website is hosted on Firebase and uses Firestore for database purposes, Cloud Run is utilized to run necessary python scripts on website.

Quick ways to run a local copy
------------------------------

Option A — super-fast (no Firebase required):

1) Serve the minimal `test-site` folder using npx (one command):

```powershell
cd C:\Users\Dave\Downloads\Thesis\Image2Web
npm run serve-test-site
# then open: http://localhost:3001
```

Option B — use the Firebase emulator (matches hosting and rewrites):

1) Ensure you have a local mapping for this repo. Use the helper:

```powershell
.\scripts\setup-firebase-local.ps1
```

or manually map a project and site for the `test` target:

```powershell
firebase login
firebase use --add            # pick a project and alias
firebase target:apply hosting test <SITE_ID>
```

2) Start the emulators (or run the single npm script):

```powershell
npm run emulators:start
# or directly: firebase emulators:start --only hosting,functions
```

The emulator will run hosting on http://localhost:5000 and functions on http://localhost:5001; hosting rewrites to `/api/**` will be routed to the test function in `test-functions/`.

Files that help automate setup:
- `scripts/setup-firebase-local.ps1` — interactive helper that creates `.firebaserc` and optionally starts the emulator.
- `.firebaserc.example` — a template you can copy and edit.

If you prefer not to install anything, open `test-site/index.html` directly in a browser (double-click) for a quick static preview.
