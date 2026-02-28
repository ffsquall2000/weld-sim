varying float vScalar;
varying vec3 vNormal;
varying vec3 vViewPosition;
attribute float scalar;

void main() {
  vScalar = scalar;
  vNormal = normalize(normalMatrix * normal);
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vViewPosition = -mvPosition.xyz;
  gl_Position = projectionMatrix * mvPosition;
}
