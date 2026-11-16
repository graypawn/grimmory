# Android reader customization

Scope: the ebook reader (EPUB and other formats rendered through Foliate), not PDF/CBX readers.

- Page navigation remains unchanged. Use the existing Tap to turn page preference to disable it; the user verified that this resolves both taps and swipes on their Android device.
- Touch taps toggle chrome without the desktop double-click delay. Chrome transitions are disabled for coarse pointers.
- EPUB reading does not automatically enter browser fullscreen. This leaves Brave and other browsers free to show their own top and bottom UI in response to their normal gestures; a webpage cannot force that browser-owned UI to appear.
- Touch text selection defaults to off. The header selection button toggles it for the current reader session; no backend or persisted preference is added. Native contextmenu events are cancelled in book documents, including while selection is enabled.

## Keeping up with upstream

Keep these changes in a small dedicated commit (or short series), separate from deployment configuration. Keep upstream's default branch unmodified and maintain a fork branch for deployment. When upgrading, merge an upstream release into the fork branch, resolve conflicts, and run the reader tests. For a clean rebuild of the fork branch, cherry-pick the customization commits onto the chosen upstream release. Do not rebase a branch shared with others without coordination.

Review upstream changes to event.service.ts and header/footer at each upgrade. Avoid replacing the reader or modifying bundled Foliate solely for these behaviors. If upstream implements a behavior, remove the corresponding local change.

## Validation

Automated regression cases live in core/event.service.spec.ts. Run from frontend with dependencies installed: `pnpm test --include=src/app/features/readers/ebook-reader/core/event.service.spec.ts`, then `pnpm typecheck`.

On Android Chrome and the installed PWA verify: with Tap to turn page disabled, every rapid edge and center tap toggles bars immediately; with it enabled, paginated navigation still works; long press does nothing by default; selection opt-in shows only Grimmory actions; turning it off clears selection; leaving and reopening the reader resets it. In a normal Brave tab, verify its browser UI can reappear with its own gestures. Also check links, highlights, chapter changes, and selection handles while scrolling. Native Android selection action modes require real-device validation; synthetic DOM tests cannot establish their behavior.
