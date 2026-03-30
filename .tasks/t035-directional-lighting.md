---
id: t035
title: "Directional lighting"
status: planned
created: 2026-03-29
updated: 2026-03-29
parent: t003
children: []
depends_on:
      - t024
blocked_by:
      - t024
      - t038
      - t039
      - t040
      - t041
      - t042
area: phoenix
---

## Context

Phoenix's PBR material infrastructure (t024) established the
material SSBO and texture slots. The next step in making rendering
useful is adding a directional light (sun) so the PBR BRDF actually
produces shaded output rather than rendering albedo-only.

## Goal

Phoenix renders objects with a directional light contribution using
the PBR BRDF (Cook-Torrance: GGX NDF, Smith G, Fresnel-Schlick).
Light direction and colour are configurable at runtime.

This task specifically targets a deferred lighting implementation: a
G-buffer pass that writes material parameters (albedo, normal, ORM,
emissive) followed by a lighting/compose pass that consumes the
G-buffer. Implement this only when a deferred renderer is available.

## Plan

- [ ] Define a `DirectionalLight` uniform struct (direction, colour,
      intensity) and upload it per-frame
- [ ] Implement the Cook-Torrance BRDF in `phoenix.slang`:
      GGX normal distribution function, Smith geometry term,
      Fresnel-Schlick approximation
- [ ] Wire albedo, normal, ORM, and emissive textures into the
      BRDF: metallic/roughness from ORM.b/g, normal from normal
      map, emissive additive on top
- [ ] Add a hardcoded sun light for initial visual validation;
      make direction/colour configurable later
- [ ] Visual validation: DamagedHelmet renders with correct
      specular highlights and diffuse shading

## Thinking

### BRDF implementation

The standard metallic-roughness Cook-Torrance split:

- **Diffuse:** Lambertian, scaled by `(1 - metallic)`.
- **Specular:** Cook-Torrance with GGX NDF, Smith correlated G,
  Fresnel-Schlick F. Roughness stored in ORM.g (green channel);
  metallic in ORM.b (blue channel) per glTF convention.

Normal mapping: transform the tangent-space normal from the normal
map into world space using the TBN matrix. Tangents need to be
present in the vertex buffer (or derived).

### Ambient term

A flat ambient term (constant × albedo) is acceptable as a
placeholder until an IBL or SSAO pass is added. It prevents
fully-lit surfaces from going completely black on the shadow side.

## Outcome

(not yet filled — task is planned)
