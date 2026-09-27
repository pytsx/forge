# Edge-driven part segmentation (v3)

The old fallback clips fixed superellipses and discards its edge flood when the
result is small. Oversized heads can therefore receive torso and arm labels.
The new `contour_rules_v3` adapter estimates a neck constriction from the silhouette,
positions body markers relative to that neck, and uses competitive watershed on
smoothed image gradients. Regions cannot overlap and no fallback replaces a small
edge region with a geometric envelope. Multiple torso markers help retain shirt
prints as part of the torso. Dependencies are already in the project.

New runs default to v3. Existing saved configurations specifying v2 still use v2.
Run segmentation again in a new run; previously generated masks are not rewritten.
The existing review/remask workflow remains necessary before reconstruction.

Optional PipelineConfig fields (API / Python):

```json
{
  "segmentation_adapter": "contour_rules_v3",
  "segmentation_split_dark_hair": true,
  "segmentation_neck_fraction": null
}
```

Dark hair splitting is disabled by default. Enable it for dark hair / lighter face
references such as Pipo. It uses upper/lower head markers and a mean intensity
contrast check, NOT a learned hair detector. Hats, highlights and dark faces can
confuse it. `segmentation_neck_fraction` overrides automatic neck estimation, as a
fraction of foreground height between 0.15 and 0.70. It applies to all views; use
aligned/cropped references and review each view. No new UI controls are added.

Limits: anatomical labels and side-view marker positions are still priors. The
method does not infer hidden surfaces, distinguish every garment/hand/sole, or
perform joint multi-view optimization. Unseeded disconnected foreground islands
can be omitted. Automatic neck detection falls back to 31% when no constriction
is supported. Confidence stays conservative and existing quality gates are kept.
This is a segmentation improvement, not manufacturing certification.

Validation: regression cases cover oversized heads in four views, exclusive masks,
a boundary away from the seed midpoint, hair opt-in, blank images and invalid
neck overrides. A resized user-supplied Pipo front reference was visually inspected;
its annotations are not ground truth and no IoU improvement is claimed. Hair/face
and neck alignment improved visibly; shoulder/sleeve separation remains approximate.

## Interface controls

Click **Gerar modelo 3D** or **Configurar e reprocessar** in the mask editor.
The execution form exposes the segmentation method, dark-hair switch, and optional
neck cut percentage (15–70%). Hair and neck controls are enabled only for v3.
Existing runs prefill their own method; choose v3 explicitly when upgrading a v2
run. New projects select v3. The mask details show `provenance.source`, while the
form separately identifies the selected run's configured method.

Submitting creates a new run through the existing API, preserving the prior run
in history. It runs the pipeline, including reconstruction, on current project
views. Unsaved mask edits must be saved first. Existing reviewed masks are not
transferred into new proposals. Printer profile and unrelated run settings are
preserved. Grounded SAM still requires server installation.
