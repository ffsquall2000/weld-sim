uniform sampler2D colormapTexture;
uniform float scalarMin;
uniform float scalarMax;
uniform float opacity;
varying float vScalar;
varying vec3 vNormal;
varying vec3 vViewPosition;

void main() {
  // Normalize scalar to [0, 1]
  float t = clamp((vScalar - scalarMin) / (scalarMax - scalarMin + 1e-10), 0.0, 1.0);

  // Sample colormap
  vec3 color = texture2D(colormapTexture, vec2(t, 0.5)).rgb;

  // Simple Lambertian shading
  vec3 lightDir = normalize(vec3(1.0, 2.0, 3.0));
  float diffuse = max(dot(normalize(vNormal), lightDir), 0.0);
  vec3 shadedColor = color * (0.3 + 0.7 * diffuse);

  gl_FragColor = vec4(shadedColor, opacity);
}
