# Browser check before final fix

Real browser upload and comparison passed, but clicking Prepare PNG download collapsed Other download options after the rerun. The PNG was ready and visible when reopened. No wrong payload: this was a visibility problem. Fixed by setting the export token in the existing button callback before rendering and keeping the ready download section expanded. Existing image algorithms/caching are unchanged.
