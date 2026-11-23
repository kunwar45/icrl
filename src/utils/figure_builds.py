# ABOUTME: Saves a matplotlib diagram as a series of "build" images, each revealing one more group of elements, for click-by-click reveals in a presentation
# ABOUTME: Use: b = FigureBuilds(ax); b.stage(1); ...draw...; b.stage(2); ...draw...; b.save(fig, out / "builds" / "name") -> name_build1.png, name_build2.png, ...
from __future__ import annotations

from pathlib import Path


class FigureBuilds:
    """Tags everything drawn on `ax` after stage(n) with n; save() writes one image per stage, cumulative.

    Stages may be revisited (stage(3), stage(5), stage(3) again), so elements that are drawn next to each
    other in the code can still appear at different clicks. Elements drawn before the first stage() call
    are always visible. Every build has the size of the full figure, so the images stack exactly.
    """

    def __init__(self, ax):
        self.ax = ax
        self.marks: list[
            tuple[int, int]
        ] = []  # (number of artists on the axes when called, stage)

    def stage(self, n: int) -> None:
        self.marks.append((len(self.ax._children), int(n)))

    def _stage_of(self, index: int) -> int:
        stage = 0
        for count, n in self.marks:
            if index >= count:
                stage = n
        return stage

    def save(self, fig, stem: Path, dpi: int = 300) -> list[Path]:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        artists = list(self.ax._children)
        written = []
        for k in sorted({n for _, n in self.marks}):
            for i, artist in enumerate(artists):
                artist.set_visible(self._stage_of(i) <= k)
            path = stem.with_name(f"{stem.name}_build{k}.png")
            fig.savefig(path, dpi=dpi, facecolor="white")
            written.append(path)
        for artist in artists:
            artist.set_visible(True)
        return written
