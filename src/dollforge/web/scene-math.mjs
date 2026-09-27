// Pixel-to-world mapping shared by image planes, masks and bounds.
export function viewFrame(view, camera, height, observations = []) {
  const fallbackAxes = {
    front: [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    back: [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    left: [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    right: [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
  };
  const calibrated = camera?.calibrated && camera.world_from_view?.length === 16
    && camera.world_units_per_pixel > 0 && camera.principal_point_px;
  if (calibrated) {
    const m = camera.world_from_view;
    return {right: [m[0], m[4], m[8]], up: [m[1], m[5], m[9]],
      normal: [m[2], m[6], m[10]], origin: [m[3], m[7], m[11]],
      scale: camera.world_units_per_pixel, principal: camera.principal_point_px, calibrated: true};
  }
  const boxes = observations.filter(o => o.review_state !== 'rejected').map(o => o.bbox_xyxy);
  const box = boxes.length ? [Math.min(...boxes.map(b => b[0])), Math.min(...boxes.map(b => b[1])),
    Math.max(...boxes.map(b => b[2])), Math.max(...boxes.map(b => b[3]))] : [0, 0, view.qa.width, view.qa.height];
  const [right, up, normal] = fallbackAxes[view.label] || fallbackAxes.front;
  return {right, up, normal, origin: [0, 0, 0], scale: height / Math.max(1, box[3] - box[1]),
    principal: [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], calibrated: false};
}

export function pixelToWorld(u, v, frame, distance = 0) {
  const x = (u - frame.principal[0]) * frame.scale;
  const y = (frame.principal[1] - v) * frame.scale;
  return frame.origin.map((a, i) => a + frame.right[i] * x + frame.up[i] * y + frame.normal[i] * distance);
}

export function maskBoundarySegments(pixels, width, height) {
  const segments = [];
  const on = (x, y) => x >= 0 && x < width && y >= 0 && y < height
    && pixels[(y * width + x) * 4] > 127;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    if (!on(x, y)) continue;
    if (!on(x, y - 1)) segments.push(x, y, x + 1, y);
    if (!on(x + 1, y)) segments.push(x + 1, y, x + 1, y + 1);
    if (!on(x, y + 1)) segments.push(x + 1, y + 1, x, y + 1);
    if (!on(x - 1, y)) segments.push(x, y + 1, x, y);
  }
  return segments;
}
