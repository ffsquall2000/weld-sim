<template>
  <div class="sim-page">
    <h1 class="sim-title">{{ $t('simulation.title') }}</h1>

    <!-- Tab Bar -->
    <div class="sim-tabs">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        class="sim-tab"
        :class="{ active: activeTab === tab.key }"
        @click="activeTab = tab.key"
      >
        <span class="tab-icon">{{ tab.icon }}</span>
        <span class="tab-label">{{ $t(tab.label) }}</span>
      </button>
    </div>

    <!-- ============ Tab 1: Stack Assembly ============ -->
    <div v-show="activeTab === 'stack'" class="sim-content">
      <div class="sim-grid-2">
        <!-- Left: Component Configuration -->
        <div class="card">
          <h2 class="card-title">{{ $t('simulation.stackConfig') }}</h2>

          <!-- Component List -->
          <div v-for="(comp, idx) in stackComponents" :key="idx" class="component-block">
            <div class="comp-header">
              <span class="comp-badge" :style="{ backgroundColor: compColor(comp.name) }">
                {{ compIcon(comp.name) }}
              </span>
              <span class="comp-name">{{ $t('simulation.comp_' + comp.name) }}</span>
              <button
                v-if="comp.name !== 'horn'"
                class="btn-icon-sm"
                @click="removeComponent(idx)"
                title="Remove"
              >✕</button>
            </div>
            <div class="comp-fields">
              <div>
                <label class="label-xs">{{ $t('simulation.hornType') }}</label>
                <select v-model="comp.horn_type" class="input-sm">
                  <option value="cylindrical">{{ $t('simulation.typeCylindrical') }}</option>
                  <option value="flat">{{ $t('simulation.typeFlat') }}</option>
                  <option value="exponential">{{ $t('simulation.typeExponential') }}</option>
                  <option value="stepped">{{ $t('simulation.typeStepped') }}</option>
                </select>
              </div>
              <div class="grid-3">
                <div>
                  <label class="label-xs">{{ $t('simulation.diameter') }}</label>
                  <input v-model.number="comp.dimensions.diameter_mm" type="number" class="input-sm" min="1" step="0.5" />
                </div>
                <div>
                  <label class="label-xs">{{ $t('simulation.length') }}</label>
                  <input v-model.number="comp.dimensions.length_mm" type="number" class="input-sm" min="1" step="0.5" />
                </div>
                <div>
                  <label class="label-xs">{{ $t('simulation.meshSize') }}</label>
                  <input v-model.number="comp.mesh_size" type="number" class="input-sm" min="0.5" step="0.5" />
                </div>
              </div>
              <div>
                <label class="label-xs">{{ $t('simulation.material') }}</label>
                <select v-model="comp.material_name" class="input-sm">
                  <option v-for="mat in assemblyMaterials" :key="mat.name" :value="mat.name">
                    {{ mat.name }}
                  </option>
                </select>
              </div>
            </div>
          </div>

          <!-- Add Component -->
          <div class="add-comp">
            <button class="btn-outline-sm" @click="addComponent('booster')">
              + {{ $t('simulation.addBooster') }}
            </button>
            <button class="btn-outline-sm" @click="addComponent('transducer')">
              + {{ $t('simulation.addTransducer') }}
            </button>
          </div>

          <!-- Analysis Settings -->
          <div class="divider"></div>
          <h3 class="subsection-title">{{ $t('simulation.analysisSettings') }}</h3>
          <div class="grid-2">
            <div>
              <label class="label-xs">{{ $t('simulation.targetFreqHz') }}</label>
              <input v-model.number="stackForm.frequency_hz" type="number" class="input-sm" min="1000" step="1000" />
            </div>
            <div>
              <label class="label-xs">{{ $t('simulation.nModes') }}</label>
              <input v-model.number="stackForm.n_modes" type="number" class="input-sm" min="1" max="50" />
            </div>
            <div>
              <label class="label-xs">{{ $t('simulation.coupling') }}</label>
              <select v-model="stackForm.coupling_method" class="input-sm">
                <option value="bonded">Bonded</option>
                <option value="penalty">Penalty</option>
              </select>
            </div>
            <div>
              <label class="label-xs">{{ $t('simulation.dampingRatio') }}</label>
              <input v-model.number="stackForm.damping_ratio" type="number" class="input-sm" min="0" max="1" step="0.001" />
            </div>
          </div>

          <button class="btn-primary mt-4" :disabled="stackLoading" @click="runStackAnalysis">
            {{ stackLoading ? $t('simulation.computing') : $t('simulation.runStack') }}
          </button>
          <FEAProgress
            ref="stackProgressRef"
            :visible="stackLoading && !!stackTaskId"
            :task-id="stackTaskId"
            :title="$t('simulation.runStack')"
            @cancel="cancelStack"
            @complete="onStackComplete"
            @error="onStackError"
          />
        </div>

        <!-- Right: Stack Results -->
        <div class="card">
          <h2 class="card-title">{{ $t('simulation.stackResults') }}</h2>

          <template v-if="stackResult">
            <!-- Key Metrics -->
            <div class="metrics-grid">
              <div class="metric-card accent">
                <div class="metric-label">{{ $t('simulation.resonanceFreq') }}</div>
                <div class="metric-value">{{ stackResult.resonance_frequency_hz.toLocaleString() }} Hz</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('simulation.gain') }}</div>
                <div class="metric-value">{{ stackResult.gain.toFixed(3) }}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('simulation.qFactor') }}</div>
                <div class="metric-value">{{ stackResult.q_factor.toFixed(1) }}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('simulation.uniformity') }}</div>
                <div class="metric-value" :style="{ color: stackResult.uniformity >= 0.9 ? '#22c55e' : stackResult.uniformity >= 0.7 ? '#eab308' : '#ef4444' }">
                  {{ (stackResult.uniformity * 100).toFixed(1) }}%
                </div>
              </div>
            </div>

            <!-- Gain Chain -->
            <div v-if="stackResult.gain_chain && Object.keys(stackResult.gain_chain).length" class="mt-4">
              <h3 class="subsection-title">{{ $t('simulation.gainChain') }}</h3>
              <div class="chain-flow">
                <template v-for="(val, key, i) in stackResult.gain_chain" :key="key">
                  <div class="chain-node">
                    <div class="chain-label">{{ key }}</div>
                    <div class="chain-val">{{ typeof val === 'number' ? val.toFixed(3) : val }}</div>
                  </div>
                  <span v-if="i < Object.keys(stackResult.gain_chain).length - 1" class="chain-arrow">→</span>
                </template>
              </div>
            </div>

            <!-- Modal Frequencies -->
            <div class="mt-4">
              <h3 class="subsection-title">{{ $t('simulation.modalFrequencies') }}</h3>
              <div class="max-h-48 overflow-y-auto">
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th class="text-right">{{ $t('simulation.freqHz') }}</th>
                      <th>{{ $t('simulation.modeType') }}</th>
                      <th class="text-center">{{ $t('simulation.deviation') }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="(freq, i) in stackResult.frequencies_hz"
                      :key="i"
                      :class="{ 'row-highlight': isClosestStackMode(freq) }"
                    >
                      <td>{{ i + 1 }}</td>
                      <td class="text-right font-mono">{{ freq.toLocaleString() }}</td>
                      <td>
                        <span class="mode-tag" :style="{ backgroundColor: modeColor(stackResult.mode_types[i] ?? '') }">
                          {{ modeLabel(stackResult.mode_types[i] ?? '') }}
                        </span>
                      </td>
                      <td class="text-center font-mono">{{ stackDeviation(freq).toFixed(1) }}%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Impedance -->
            <div v-if="stackResult.impedance && Object.keys(stackResult.impedance).length" class="mt-4">
              <h3 class="subsection-title">{{ $t('simulation.impedance') }}</h3>
              <div class="grid-2 text-sm">
                <div v-for="(val, key) in stackResult.impedance" :key="key" class="kv-item">
                  <span class="kv-key">{{ key }}</span>
                  <span class="kv-val font-mono">{{ typeof val === 'number' ? val.toFixed(2) : val }}</span>
                </div>
              </div>
            </div>

            <div class="solve-info">
              {{ $t('simulation.solveTime') }}: {{ stackResult.solve_time_s.toFixed(2) }}s
              · DOF: {{ stackResult.n_total_dof.toLocaleString() }}
              · {{ $t('simulation.components') }}: {{ stackResult.n_components }}
            </div>
          </template>

          <div v-else-if="stackError" class="error-box">{{ stackError }}</div>
          <div v-else class="placeholder-text">{{ $t('simulation.stackPlaceholder') }}</div>
        </div>
      </div>
    </div>

    <!-- ============ Tab 2: Modal / FEA Analysis ============ -->
    <div v-show="activeTab === 'modal'" class="sim-content">
      <div class="sim-grid-2">
        <!-- Left: FEA Parameters -->
        <div class="card">
          <h2 class="card-title">{{ $t('simulation.modalParams') }}</h2>

          <div class="space-y-3">
            <div>
              <label class="label-xs">{{ $t('simulation.hornType') }}</label>
              <select v-model="feaForm.horn_type" class="input-sm">
                <option value="cylindrical">{{ $t('simulation.typeCylindrical') }}</option>
                <option value="flat">{{ $t('simulation.typeFlat') }}</option>
                <option value="exponential">{{ $t('simulation.typeExponential') }}</option>
                <option value="blade">{{ $t('simulation.typeBlade') }}</option>
                <option value="block">{{ $t('simulation.typeBlock') }}</option>
              </select>
            </div>
            <div class="grid-3">
              <div>
                <label class="label-xs">{{ $t('simulation.widthMm') }}</label>
                <input v-model.number="feaForm.width_mm" type="number" class="input-sm" min="1" step="0.5" />
              </div>
              <div>
                <label class="label-xs">{{ $t('simulation.heightMm') }}</label>
                <input v-model.number="feaForm.height_mm" type="number" class="input-sm" min="1" step="0.5" />
              </div>
              <div>
                <label class="label-xs">{{ $t('simulation.lengthMm') }}</label>
                <input v-model.number="feaForm.length_mm" type="number" class="input-sm" min="1" step="0.5" />
              </div>
            </div>
            <div>
              <label class="label-xs">{{ $t('simulation.material') }}</label>
              <select v-model="feaForm.material" class="input-sm">
                <option v-for="mat in feaMaterials" :key="mat.name" :value="mat.name">
                  {{ mat.name }} ({{ mat.E_gpa }} GPa)
                </option>
              </select>
            </div>
            <div class="grid-2">
              <div>
                <label class="label-xs">{{ $t('simulation.targetFreqKhz') }}</label>
                <input v-model.number="feaForm.frequency_khz" type="number" class="input-sm" min="1" step="0.5" />
              </div>
              <div>
                <label class="label-xs">{{ $t('simulation.meshDensity') }}</label>
                <select v-model="feaForm.mesh_density" class="input-sm">
                  <option value="coarse">{{ $t('simulation.meshCoarse') }}</option>
                  <option value="medium">{{ $t('simulation.meshMedium') }}</option>
                  <option value="fine">{{ $t('simulation.meshFine') }}</option>
                </select>
              </div>
            </div>
          </div>

          <!-- Upload STEP file -->
          <div class="divider"></div>
          <div class="upload-zone" @click="feaFileInput?.click()" @dragover.prevent @drop.prevent="onFEAFileDrop">
            <div v-if="feaStepFile" class="upload-info">
              <span class="file-icon">📐</span>
              <span>{{ feaStepFile.name }}</span>
              <button class="btn-icon-sm" @click.stop="feaStepFile = null">✕</button>
            </div>
            <div v-else class="upload-hint">
              <span>📁 {{ $t('simulation.dropStepHint') }}</span>
            </div>
            <input ref="feaFileInput" type="file" accept=".step,.stp" class="hidden" @change="onFEAFileSelect" />
          </div>
          <div v-if="feaStepFile" class="cad-badge">
            <span class="badge-dot"></span> {{ $t('simulation.usingCadGeometry') }}
          </div>

          <button class="btn-primary mt-4" :disabled="feaLoading" @click="runFEA">
            {{ feaLoading ? $t('simulation.computing') : $t('simulation.runModal') }}
          </button>
          <FEAProgress
            ref="feaProgressRef"
            :visible="feaLoading && !!feaTaskId"
            :task-id="feaTaskId"
            :title="$t('simulation.runModal')"
            @cancel="cancelFEA"
            @complete="onFEAComplete"
            @error="onFEAError"
          />
        </div>

        <!-- Right: FEA Results -->
        <div class="card">
          <h2 class="card-title">{{ $t('simulation.modalResults') }}</h2>

          <template v-if="feaResult">
            <!-- Summary Metrics -->
            <div class="metrics-grid">
              <div class="metric-card accent">
                <div class="metric-label">{{ $t('simulation.closestMode') }}</div>
                <div class="metric-value">{{ feaResult.closest_mode_hz.toLocaleString() }} Hz</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('simulation.deviation') }}</div>
                <div class="metric-value" :style="{ color: feaDeviationColor }">
                  {{ feaResult.frequency_deviation_percent.toFixed(2) }}%
                </div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('simulation.maxStress') }}</div>
                <div class="metric-value">{{ feaResult.stress_max_mpa?.toFixed(1) ?? '--' }} MPa</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('simulation.targetFreq') }}</div>
                <div class="metric-value">{{ feaResult.target_frequency_hz.toLocaleString() }} Hz</div>
              </div>
            </div>

            <!-- Modal Bar Chart -->
            <div v-if="feaResult.mode_shapes.length" class="mt-4">
              <h3 class="subsection-title">{{ $t('simulation.resonantModes') }}</h3>
              <ModalBarChart
                :modes="feaResult.mode_shapes.map((m, i) => ({ modeNumber: i + 1, frequency: m.frequency_hz, type: m.mode_type as any }))"
                :target-frequency="feaResult.target_frequency_hz"
                style="height: 200px"
              />
            </div>

            <!-- Mode Table -->
            <div class="mt-4">
              <div class="max-h-48 overflow-y-auto">
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th class="text-right">{{ $t('simulation.freqHz') }}</th>
                      <th>{{ $t('simulation.modeType') }}</th>
                      <th class="text-center">{{ $t('simulation.deviation') }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="(mode, i) in feaResult.mode_shapes"
                      :key="i"
                      :class="{ 'row-highlight': Math.abs(mode.frequency_hz - feaResult!.closest_mode_hz) < 1 }"
                    >
                      <td>{{ i + 1 }}</td>
                      <td class="text-right font-mono">{{ mode.frequency_hz.toLocaleString() }}</td>
                      <td>
                        <span class="mode-tag" :style="{ backgroundColor: modeColor(mode.mode_type) }">
                          {{ modeLabel(mode.mode_type) }}
                        </span>
                      </td>
                      <td class="text-center font-mono">
                        {{ (Math.abs(mode.frequency_hz - feaResult!.target_frequency_hz) / feaResult!.target_frequency_hz * 100).toFixed(1) }}%
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div class="solve-info">
              {{ $t('simulation.solveTime') }}: {{ feaResult.solve_time_s.toFixed(2) }}s
              · {{ $t('simulation.nodes') }}: {{ feaResult.node_count.toLocaleString() }}
              · {{ $t('simulation.elements') }}: {{ feaResult.element_count.toLocaleString() }}
            </div>
          </template>

          <div v-else-if="feaError" class="error-box">{{ feaError }}</div>
          <div v-else class="placeholder-text">{{ $t('simulation.modalPlaceholder') }}</div>
        </div>
      </div>
    </div>

    <!-- ============ Tab 3: Acoustic Analysis ============ -->
    <div v-show="activeTab === 'acoustic'" class="sim-content">
      <div class="sim-grid-2">
        <!-- Left: Acoustic Parameters -->
        <div class="card">
          <h2 class="card-title">{{ $t('simulation.acousticParams') }}</h2>
          <div class="space-y-3">
            <div>
              <label class="label-xs">{{ $t('simulation.hornType') }}</label>
              <select v-model="acousticForm.horn_type" class="input-sm">
                <option value="cylindrical">{{ $t('simulation.typeCylindrical') }}</option>
                <option value="flat">{{ $t('simulation.typeFlat') }}</option>
                <option value="exponential">{{ $t('simulation.typeExponential') }}</option>
                <option value="blade">{{ $t('simulation.typeBlade') }}</option>
                <option value="stepped">{{ $t('simulation.typeStepped') }}</option>
                <option value="block">{{ $t('simulation.typeBlock') }}</option>
              </select>
            </div>
            <div class="grid-3">
              <div>
                <label class="label-xs">{{ $t('simulation.widthMm') }}</label>
                <input v-model.number="acousticForm.width_mm" type="number" class="input-sm" min="1" step="0.5" />
              </div>
              <div>
                <label class="label-xs">{{ $t('simulation.heightMm') }}</label>
                <input v-model.number="acousticForm.height_mm" type="number" class="input-sm" min="1" step="0.5" />
              </div>
              <div>
                <label class="label-xs">{{ $t('simulation.lengthMm') }}</label>
                <input v-model.number="acousticForm.length_mm" type="number" class="input-sm" min="1" step="0.5" />
              </div>
            </div>
            <div>
              <label class="label-xs">{{ $t('simulation.material') }}</label>
              <select v-model="acousticForm.material" class="input-sm">
                <option v-for="mat in feaMaterials" :key="mat.name" :value="mat.name">
                  {{ mat.name }} ({{ mat.E_gpa }} GPa)
                </option>
              </select>
            </div>
            <div class="grid-2">
              <div>
                <label class="label-xs">{{ $t('simulation.targetFreqKhz') }}</label>
                <input v-model.number="acousticForm.frequency_khz" type="number" class="input-sm" min="1" step="0.5" />
              </div>
              <div>
                <label class="label-xs">{{ $t('simulation.meshDensity') }}</label>
                <select v-model="acousticForm.mesh_density" class="input-sm">
                  <option value="coarse">{{ $t('simulation.meshCoarse') }}</option>
                  <option value="medium">{{ $t('simulation.meshMedium') }}</option>
                  <option value="fine">{{ $t('simulation.meshFine') }}</option>
                </select>
              </div>
            </div>
          </div>
          <button class="btn-primary mt-4" :disabled="acousticLoading" @click="runAcoustic">
            {{ acousticLoading ? $t('simulation.computing') : $t('simulation.runAcoustic') }}
          </button>
          <FEAProgress
            ref="acousticProgressRef"
            :visible="acousticLoading && !!acousticTaskId"
            :task-id="acousticTaskId"
            :title="$t('simulation.runAcoustic')"
            @cancel="cancelAcoustic"
            @complete="onAcousticComplete"
            @error="onAcousticError"
          />
        </div>

        <!-- Right: Acoustic Results -->
        <div class="card">
          <h2 class="card-title">{{ $t('simulation.acousticResults') }}</h2>

          <template v-if="acousticResult">
            <!-- Key Metrics -->
            <div class="metrics-grid">
              <div class="metric-card accent">
                <div class="metric-label">{{ $t('simulation.closestMode') }}</div>
                <div class="metric-value">{{ acousticResult.closest_mode_hz.toLocaleString() }} Hz</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('simulation.amplitudeUniformity') }}</div>
                <div class="metric-value" :style="{ color: acousticResult.amplitude_uniformity >= 0.9 ? '#22c55e' : acousticResult.amplitude_uniformity >= 0.7 ? '#eab308' : '#ef4444' }">
                  {{ (acousticResult.amplitude_uniformity * 100).toFixed(1) }}%
                </div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('simulation.maxStress') }}</div>
                <div class="metric-value">{{ acousticResult.stress_max_mpa?.toFixed(1) ?? '--' }} MPa</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('simulation.freqDeviation') }}</div>
                <div class="metric-value">{{ acousticResult.frequency_deviation_percent?.toFixed(2) }}%</div>
              </div>
            </div>

            <!-- Amplitude Uniformity Bar -->
            <div class="mt-4">
              <h3 class="subsection-title">{{ $t('simulation.amplitudeUniformity') }}</h3>
              <div class="uniformity-bar">
                <div
                  class="uniformity-fill"
                  :style="{
                    width: (acousticResult.amplitude_uniformity * 100) + '%',
                    backgroundColor: acousticResult.amplitude_uniformity >= 0.9 ? '#22c55e' : acousticResult.amplitude_uniformity >= 0.7 ? '#eab308' : '#ef4444'
                  }"
                ></div>
              </div>
              <div class="uniformity-label" :style="{ color: acousticResult.amplitude_uniformity >= 0.9 ? '#22c55e' : acousticResult.amplitude_uniformity >= 0.7 ? '#eab308' : '#ef4444' }">
                {{ acousticResult.amplitude_uniformity >= 0.9 ? $t('simulation.uniformityExcellent') : acousticResult.amplitude_uniformity >= 0.7 ? $t('simulation.uniformityAcceptable') : $t('simulation.uniformityPoor') }}
              </div>
            </div>

            <!-- FRF Chart -->
            <div v-if="acousticResult.harmonic_response" class="mt-4">
              <h3 class="subsection-title">{{ $t('simulation.frfChart') }}</h3>
              <FRFChart
                :frequencies="acousticResult.harmonic_response.frequencies_hz"
                :amplitudes="acousticResult.harmonic_response.amplitudes"
                :target-frequency="acousticResult.target_frequency_hz"
                :peak-frequency="acousticPeakFreq ?? undefined"
                :peak-amplitude="acousticPeakAmp ?? undefined"
              />
            </div>

            <!-- Stress Hotspots -->
            <div v-if="acousticResult.stress_hotspots?.length" class="mt-4">
              <h3 class="subsection-title">{{ $t('simulation.stressHotspots') }}</h3>
              <div class="max-h-40 overflow-y-auto">
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>{{ $t('simulation.location') }}</th>
                      <th class="text-right">{{ $t('simulation.stressMpa') }}</th>
                      <th class="text-center">{{ $t('simulation.severity') }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(hs, i) in acousticResult.stress_hotspots" :key="i">
                      <td class="font-mono">({{ hs.location.map((v: number) => v.toFixed(1)).join(', ') }})</td>
                      <td class="text-right font-mono">{{ hs.von_mises_mpa?.toFixed(1) }}</td>
                      <td class="text-center">
                        <span class="severity-tag" :style="{ backgroundColor: severityColor(hotspotSeverity(hs, acousticResult!.stress_max_mpa)) }">
                          {{ hotspotSeverity(hs, acousticResult!.stress_max_mpa) }}
                        </span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Mode Table -->
            <div v-if="acousticResult.modes?.length" class="mt-4">
              <h3 class="subsection-title">{{ $t('simulation.modalFrequencies') }}</h3>
              <ModalBarChart
                :modes="acousticResult.modes.map((m: any, i: number) => ({ modeNumber: i + 1, frequency: m.frequency_hz, type: m.mode_type }))"
                :target-frequency="acousticResult.target_frequency_hz"
                style="height: 200px"
              />
            </div>

            <div class="solve-info">
              {{ $t('simulation.solveTime') }}: {{ acousticResult.solve_time_s.toFixed(2) }}s
              · {{ $t('simulation.nodes') }}: {{ acousticResult.node_count.toLocaleString() }}
              · {{ $t('simulation.elements') }}: {{ acousticResult.element_count.toLocaleString() }}
            </div>
          </template>

          <div v-else-if="acousticError" class="error-box">{{ acousticError }}</div>
          <div v-else class="placeholder-text">{{ $t('simulation.acousticPlaceholder') }}</div>
        </div>
      </div>
    </div>

    <!-- ============ Tab 4: Welding Process Simulation ============ -->
    <div v-show="activeTab === 'weld'" class="sim-content">
      <div class="sim-grid-2">
        <!-- Left: Weld Parameters -->
        <div class="card">
          <h2 class="card-title">{{ $t('simulation.weldParams') }}</h2>

          <div class="space-y-3">
            <!-- Application -->
            <div>
              <label class="label-xs">{{ $t('simulation.application') }}</label>
              <select v-model="weldForm.application" class="input-sm">
                <option value="li_battery_tab">{{ $t('simulation.appLiBatteryTab') }}</option>
                <option value="li_battery_busbar">{{ $t('simulation.appLiBatteryBusbar') }}</option>
                <option value="li_battery_collector">{{ $t('simulation.appLiBatteryCollector') }}</option>
                <option value="general_metal">{{ $t('simulation.appGeneralMetal') }}</option>
              </select>
            </div>

            <!-- Materials -->
            <div class="divider"></div>
            <h3 class="subsection-title">{{ $t('simulation.materialConfig') }}</h3>
            <div class="grid-2">
              <div>
                <label class="label-xs">{{ $t('simulation.upperMaterial') }}</label>
                <select v-model="weldForm.upper_material_type" class="input-sm">
                  <option v-for="m in weldMaterials" :key="m" :value="m">{{ m }}</option>
                </select>
              </div>
              <div>
                <label class="label-xs">{{ $t('simulation.lowerMaterial') }}</label>
                <select v-model="weldForm.lower_material_type" class="input-sm">
                  <option v-for="m in weldMaterials" :key="m" :value="m">{{ m }}</option>
                </select>
              </div>
            </div>
            <div class="grid-3">
              <div>
                <label class="label-xs">{{ $t('simulation.upperThickness') }}</label>
                <input v-model.number="weldForm.upper_thickness_mm" type="number" class="input-sm" min="0.001" step="0.001" />
              </div>
              <div>
                <label class="label-xs">{{ $t('simulation.layers') }}</label>
                <input v-model.number="weldForm.upper_layers" type="number" class="input-sm" min="1" max="200" />
              </div>
              <div>
                <label class="label-xs">{{ $t('simulation.lowerThickness') }}</label>
                <input v-model.number="weldForm.lower_thickness_mm" type="number" class="input-sm" min="0.001" step="0.01" />
              </div>
            </div>

            <!-- Weld Geometry -->
            <div class="divider"></div>
            <h3 class="subsection-title">{{ $t('simulation.weldGeometry') }}</h3>
            <div class="grid-2">
              <div>
                <label class="label-xs">{{ $t('simulation.weldWidth') }}</label>
                <input v-model.number="weldForm.weld_width_mm" type="number" class="input-sm" min="0.1" step="0.5" />
              </div>
              <div>
                <label class="label-xs">{{ $t('simulation.weldLength') }}</label>
                <input v-model.number="weldForm.weld_length_mm" type="number" class="input-sm" min="0.1" step="0.5" />
              </div>
            </div>

            <!-- Equipment -->
            <div class="divider"></div>
            <h3 class="subsection-title">{{ $t('simulation.equipment') }}</h3>
            <div class="grid-3">
              <div>
                <label class="label-xs">{{ $t('simulation.freqKhz') }}</label>
                <input v-model.number="weldForm.frequency_khz" type="number" class="input-sm" min="1" step="0.5" />
              </div>
              <div>
                <label class="label-xs">{{ $t('simulation.maxPowerW') }}</label>
                <input v-model.number="weldForm.max_power_w" type="number" class="input-sm" min="100" step="100" />
              </div>
              <div>
                <label class="label-xs">{{ $t('simulation.boosterGain') }}</label>
                <input v-model.number="weldForm.booster_gain" type="number" class="input-sm" min="0.1" step="0.1" />
              </div>
            </div>
          </div>

          <button class="btn-primary mt-4" :disabled="weldLoading" @click="runWeldSimulation">
            {{ weldLoading ? $t('simulation.computing') : $t('simulation.runWeld') }}
          </button>
        </div>

        <!-- Right: Weld Results -->
        <div class="card">
          <h2 class="card-title">{{ $t('simulation.weldResults') }}</h2>

          <template v-if="weldResult">
            <!-- Recipe ID -->
            <div class="recipe-id">
              {{ $t('simulation.recipeId') }}: <span class="font-mono">{{ weldResult.recipe_id }}</span>
            </div>

            <!-- Key Parameters -->
            <div class="metrics-grid mt-3">
              <div class="metric-card accent">
                <div class="metric-label">{{ $t('simulation.amplitude') }}</div>
                <div class="metric-value">{{ weldResult.parameters?.amplitude_um?.toFixed(1) ?? '--' }} μm</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('simulation.pressure') }}</div>
                <div class="metric-value">{{ weldResult.parameters?.pressure_mpa?.toFixed(2) ?? '--' }} MPa</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('simulation.energy') }}</div>
                <div class="metric-value">{{ weldResult.parameters?.energy_j?.toFixed(1) ?? '--' }} J</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('simulation.weldTime') }}</div>
                <div class="metric-value">{{ weldResult.parameters?.weld_time_s?.toFixed(3) ?? '--' }} s</div>
              </div>
            </div>

            <!-- Safety Window -->
            <div v-if="weldResult.safety_window" class="mt-4">
              <h3 class="subsection-title">{{ $t('simulation.safetyWindow') }}</h3>
              <div class="max-h-48 overflow-y-auto">
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>{{ $t('simulation.parameter') }}</th>
                      <th class="text-right">{{ $t('simulation.value') }}</th>
                      <th class="text-right">{{ $t('simulation.safeMin') }}</th>
                      <th class="text-right">{{ $t('simulation.safeMax') }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(sw, key) in weldResult.safety_window" :key="key">
                      <td>{{ key }}</td>
                      <td class="text-right font-mono">{{ typeof sw.value === 'number' ? sw.value.toFixed(2) : sw.value }}</td>
                      <td class="text-right font-mono">{{ typeof sw.min === 'number' ? sw.min.toFixed(2) : sw.min }}</td>
                      <td class="text-right font-mono">{{ typeof sw.max === 'number' ? sw.max.toFixed(2) : sw.max }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Risk Assessment -->
            <div v-if="weldResult.risk_assessment" class="mt-4">
              <h3 class="subsection-title">{{ $t('simulation.riskAssessment') }}</h3>
              <div class="grid-2 text-sm">
                <div v-for="(val, key) in weldResult.risk_assessment" :key="key" class="kv-item">
                  <span class="kv-key">{{ key }}</span>
                  <span class="kv-val font-mono" :style="{ color: riskColor(val) }">
                    {{ typeof val === 'number' ? val.toFixed(2) : val }}
                  </span>
                </div>
              </div>
            </div>

            <!-- Recommendations -->
            <div v-if="weldResult.recommendations?.length" class="mt-4">
              <h3 class="subsection-title">{{ $t('simulation.recommendations') }}</h3>
              <ul class="rec-list">
                <li v-for="(rec, i) in weldResult.recommendations" :key="i">{{ rec }}</li>
              </ul>
            </div>

            <!-- Validation -->
            <div v-if="weldResult.validation" class="mt-4">
              <div
                class="validation-badge"
                :class="weldResult.validation.status === 'pass' ? 'pass' : 'fail'"
              >
                {{ weldResult.validation.status === 'pass' ? '✓ PASS' : '✕ FAIL' }}
              </div>
              <ul v-if="weldResult.validation.messages?.length" class="rec-list mt-2">
                <li v-for="(msg, i) in weldResult.validation.messages" :key="i">{{ msg }}</li>
              </ul>
            </div>
          </template>

          <div v-else-if="weldError" class="error-box">{{ weldError }}</div>
          <div v-else class="placeholder-text">{{ $t('simulation.weldPlaceholder') }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import apiClient from '@/api/client'
import { geometryApi, type FEAResponse, type FEAMaterial } from '@/api/geometry'
import { assemblyApi, type AssemblyAnalysisResponse, type AssemblyMaterial, type ComponentRequest } from '@/api/assembly'
import ModalBarChart from '@/components/charts/ModalBarChart.vue'
import FRFChart from '@/components/charts/FRFChart.vue'
import FEAProgress from '@/components/FEAProgress.vue'
import { generateTaskId } from '@/utils/uuid'

const { t } = useI18n()

// ---------- Tab Management ----------
const activeTab = ref<'stack' | 'modal' | 'acoustic' | 'weld'>('stack')
const tabs = [
  { key: 'stack' as const, icon: '🔧', label: 'simulation.tabStack' },
  { key: 'modal' as const, icon: '📊', label: 'simulation.tabModal' },
  { key: 'acoustic' as const, icon: '🔊', label: 'simulation.tabAcoustic' },
  { key: 'weld' as const, icon: '⚡', label: 'simulation.tabWeld' },
]

// ---------- Shared ----------
const feaMaterials = ref<FEAMaterial[]>([
  { name: 'Titanium Ti-6Al-4V', E_gpa: 113.8, density_kg_m3: 4430, poisson_ratio: 0.342 },
  { name: 'Steel D2', E_gpa: 210, density_kg_m3: 7700, poisson_ratio: 0.3 },
  { name: 'Aluminum 7075-T6', E_gpa: 71.7, density_kg_m3: 2810, poisson_ratio: 0.33 },
  { name: 'Copper C11000', E_gpa: 117, density_kg_m3: 8940, poisson_ratio: 0.34 },
  { name: 'Nickel 200', E_gpa: 204, density_kg_m3: 8890, poisson_ratio: 0.31 },
  { name: 'M2 High Speed Steel', E_gpa: 220, density_kg_m3: 8160, poisson_ratio: 0.27 },
  { name: 'CPM 10V', E_gpa: 222, density_kg_m3: 7640, poisson_ratio: 0.28 },
  { name: 'PM60 Powder Steel', E_gpa: 230, density_kg_m3: 8050, poisson_ratio: 0.28 },
  { name: 'HAP40 Powder HSS', E_gpa: 228, density_kg_m3: 8100, poisson_ratio: 0.28 },
  { name: 'HAP72 Powder HSS', E_gpa: 235, density_kg_m3: 8200, poisson_ratio: 0.28 },
])
const assemblyMaterials = ref<AssemblyMaterial[]>([])
const weldMaterials = ref<string[]>(['Cu', 'Al', 'Ni', 'Steel', 'Ti'])

function modeColor(type: string): string {
  switch (type) {
    case 'longitudinal': return '#22c55e'
    case 'flexural': return '#3b82f6'
    case 'torsional': return '#a855f7'
    default: return '#6b7280'
  }
}

function modeLabel(type: string): string {
  switch (type) {
    case 'longitudinal': return t('simulation.modeLongitudinal')
    case 'flexural': return t('simulation.modeFlexural')
    case 'torsional': return t('simulation.modeTorsional')
    default: return type
  }
}

function severityColor(s: string): string {
  return s === 'high' ? '#ef4444' : s === 'medium' ? '#eab308' : '#22c55e'
}

function hotspotSeverity(hs: any, maxStress: number): string {
  if (maxStress <= 0) return 'low'
  const ratio = (hs.von_mises_mpa ?? 0) / maxStress
  return ratio > 0.8 ? 'high' : ratio > 0.5 ? 'medium' : 'low'
}

function riskColor(val: any): string {
  if (typeof val === 'string') {
    if (val === 'high' || val === 'critical') return '#ef4444'
    if (val === 'medium') return '#eab308'
    return '#22c55e'
  }
  if (typeof val === 'number') {
    if (val > 0.8) return '#ef4444'
    if (val > 0.5) return '#eab308'
    return '#22c55e'
  }
  return ''
}

// ---------- Stack Assembly ----------
const stackComponents = ref<ComponentRequest[]>([
  {
    name: 'horn',
    horn_type: 'cylindrical',
    dimensions: { diameter_mm: 50, length_mm: 80 },
    material_name: 'Titanium Ti-6Al-4V',
    mesh_size: 2.0,
  },
  {
    name: 'booster',
    horn_type: 'cylindrical',
    dimensions: { diameter_mm: 50, length_mm: 60 },
    material_name: 'Titanium Ti-6Al-4V',
    mesh_size: 2.0,
  },
])

const stackForm = ref({
  frequency_hz: 20000,
  n_modes: 20,
  coupling_method: 'bonded',
  damping_ratio: 0.005,
})

const stackResult = ref<AssemblyAnalysisResponse | null>(null)
const stackLoading = ref(false)
const stackError = ref<string | null>(null)
const stackTaskId = ref('')
const stackProgressRef = ref<InstanceType<typeof FEAProgress> | null>(null)

function compColor(name: string): string {
  switch (name) {
    case 'horn': return '#ea580c'
    case 'booster': return '#2563eb'
    case 'transducer': return '#7c3aed'
    default: return '#6b7280'
  }
}

function compIcon(name: string): string {
  switch (name) {
    case 'horn': return 'H'
    case 'booster': return 'B'
    case 'transducer': return 'T'
    default: return '?'
  }
}

function addComponent(name: string) {
  stackComponents.value.push({
    name,
    horn_type: 'cylindrical',
    dimensions: { diameter_mm: 50, length_mm: name === 'transducer' ? 40 : 60 },
    material_name: 'Titanium Ti-6Al-4V',
    mesh_size: 2.0,
  })
}

function removeComponent(idx: number) {
  stackComponents.value.splice(idx, 1)
}

function isClosestStackMode(freq: number): boolean {
  if (!stackResult.value) return false
  return Math.abs(freq - stackResult.value.resonance_frequency_hz) < 1
}

function stackDeviation(freq: number): number {
  if (!stackResult.value || stackForm.value.frequency_hz <= 0) return 0
  return Math.abs(freq - stackForm.value.frequency_hz) / stackForm.value.frequency_hz * 100
}

async function runStackAnalysis() {
  stackLoading.value = true
  stackError.value = null
  stackResult.value = null
  // Generate task_id BEFORE the API call so FEAProgress can connect WebSocket immediately
  const tid = generateTaskId()
  stackTaskId.value = tid
  try {
    const req = {
      components: stackComponents.value,
      coupling_method: stackForm.value.coupling_method,
      penalty_factor: 1e3,
      analyses: ['modal', 'harmonic'],
      frequency_hz: stackForm.value.frequency_hz,
      n_modes: stackForm.value.n_modes,
      damping_ratio: stackForm.value.damping_ratio,
      use_gmsh: true,
      task_id: tid,
    }
    const res = await assemblyApi.analyze(req)
    stackResult.value = res.data
  } catch (err: any) {
    stackError.value = err.response?.data?.detail || err.message || 'Stack analysis failed'
  } finally {
    stackLoading.value = false
  }
}

function cancelStack() {
  stackProgressRef.value?.requestCancel()
}

function onStackComplete() {
  stackLoading.value = false
}

function onStackError(error: string) {
  stackError.value = error
  stackLoading.value = false
}

// ---------- Modal / FEA ----------
const feaForm = ref({
  horn_type: 'cylindrical',
  width_mm: 25,
  height_mm: 80,
  length_mm: 25,
  material: 'Titanium Ti-6Al-4V',
  frequency_khz: 20,
  mesh_density: 'medium',
})
const feaStepFile = ref<File | null>(null)
const feaFileInput = ref<HTMLInputElement | null>(null)
const feaResult = ref<FEAResponse | null>(null)
const feaLoading = ref(false)
const feaError = ref<string | null>(null)
const feaTaskId = ref('')
const feaProgressRef = ref<InstanceType<typeof FEAProgress> | null>(null)

const feaDeviationColor = computed(() => {
  if (!feaResult.value) return ''
  const d = Math.abs(feaResult.value.frequency_deviation_percent)
  if (d < 2) return '#22c55e'
  if (d < 5) return '#eab308'
  return '#ef4444'
})

function onFEAFileSelect(e: Event) {
  const input = e.target as HTMLInputElement
  const f = input.files?.[0]
  if (f) feaStepFile.value = f
}

function onFEAFileDrop(e: DragEvent) {
  const file = e.dataTransfer?.files[0]
  if (file) {
    const ext = file.name.split('.').pop()?.toLowerCase()
    if (ext === 'step' || ext === 'stp') feaStepFile.value = file
  }
}

async function runFEA() {
  feaLoading.value = true
  feaError.value = null
  feaResult.value = null
  // Generate task_id BEFORE the API call so FEAProgress can connect WebSocket immediately
  const tid = generateTaskId()
  feaTaskId.value = tid
  try {
    let res
    if (feaStepFile.value) {
      res = await geometryApi.runFEAOnStep(
        feaStepFile.value,
        feaForm.value.material,
        feaForm.value.frequency_khz,
        feaForm.value.mesh_density,
        tid,
      )
    } else {
      res = await geometryApi.runFEA({ ...feaForm.value, task_id: tid })
    }
    feaResult.value = res.data
  } catch (err: any) {
    feaError.value = err.response?.data?.detail || err.message || 'FEA failed'
  } finally {
    feaLoading.value = false
  }
}

function cancelFEA() {
  feaProgressRef.value?.requestCancel()
}

function onFEAComplete() {
  feaLoading.value = false
}

function onFEAError(error: string) {
  feaError.value = error
  feaLoading.value = false
}

// ---------- Acoustic ----------
const acousticForm = ref({
  horn_type: 'cylindrical',
  width_mm: 25,
  height_mm: 80,
  length_mm: 25,
  material: 'Titanium Ti-6Al-4V',
  frequency_khz: 20,
  mesh_density: 'medium',
})
const acousticResult = ref<any>(null)
const acousticLoading = ref(false)
const acousticError = ref<string | null>(null)
const acousticTaskId = ref('')
const acousticProgressRef = ref<InstanceType<typeof FEAProgress> | null>(null)

const acousticPeakAmp = computed(() => {
  if (!acousticResult.value?.harmonic_response) return null
  return Math.max(...acousticResult.value.harmonic_response.amplitudes)
})

const acousticPeakFreq = computed(() => {
  if (!acousticResult.value?.harmonic_response) return null
  const amps = acousticResult.value.harmonic_response.amplitudes
  const freqs = acousticResult.value.harmonic_response.frequencies_hz
  return freqs[amps.indexOf(Math.max(...amps))]
})

async function runAcoustic() {
  acousticLoading.value = true
  acousticError.value = null
  acousticResult.value = null
  // Generate task_id BEFORE the API call so FEAProgress can connect WebSocket immediately
  const tid = generateTaskId()
  acousticTaskId.value = tid
  try {
    const res = await apiClient.post('/acoustic/analyze', { ...acousticForm.value, task_id: tid }, { timeout: 360000 })
    acousticResult.value = res.data
  } catch (err: any) {
    acousticError.value = err.response?.data?.detail || err.message || 'Acoustic analysis failed'
  } finally {
    acousticLoading.value = false
  }
}

function cancelAcoustic() {
  acousticProgressRef.value?.requestCancel()
}

function onAcousticComplete() {
  acousticLoading.value = false
}

function onAcousticError(error: string) {
  acousticError.value = error
  acousticLoading.value = false
}

// ---------- Welding Process ----------
const weldForm = ref({
  application: 'li_battery_tab',
  upper_material_type: 'Cu',
  upper_thickness_mm: 0.008,
  upper_layers: 40,
  lower_material_type: 'Ni',
  lower_thickness_mm: 0.2,
  weld_width_mm: 5,
  weld_length_mm: 40,
  frequency_khz: 20,
  max_power_w: 3500,
  booster_gain: 1.5,
})
const weldResult = ref<any>(null)
const weldLoading = ref(false)
const weldError = ref<string | null>(null)

async function runWeldSimulation() {
  weldLoading.value = true
  weldError.value = null
  weldResult.value = null
  try {
    const res = await apiClient.post('/simulate', weldForm.value, { timeout: 60000 })
    weldResult.value = res.data
  } catch (err: any) {
    weldError.value = err.response?.data?.detail || err.message || 'Simulation failed'
  } finally {
    weldLoading.value = false
  }
}

// ---------- Initialize ----------
onMounted(async () => {
  // Load materials
  try {
    const [feaRes, matRes] = await Promise.allSettled([
      geometryApi.getMaterials(),
      apiClient.get<{ materials: string[] }>('/materials'),
    ])
    if (feaRes.status === 'fulfilled') feaMaterials.value = feaRes.value.data
    if (matRes.status === 'fulfilled') weldMaterials.value = matRes.value.data.materials
  } catch { /* use defaults */ }

  try {
    const asmRes = await assemblyApi.getMaterials()
    assemblyMaterials.value = asmRes.data
  } catch {
    // Fallback: use feaMaterials as assembly materials
    assemblyMaterials.value = feaMaterials.value.map(m => ({
      ...m,
      acoustic_velocity_m_s: null,
    }))
  }
})
</script>

<style scoped>
.sim-page {
  padding: 1.5rem;
  max-width: 1400px;
  margin: 0 auto;
}

.sim-title {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
  color: var(--color-text-primary);
}

/* Tabs */
.sim-tabs {
  display: flex;
  gap: 0;
  border-bottom: 2px solid var(--color-border);
  margin-bottom: 1.25rem;
}

.sim-tab {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 20px;
  border: none;
  background: none;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--color-text-secondary);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
  transition: color 0.15s, border-color 0.15s;
}

.sim-tab:hover {
  color: var(--color-text-primary);
}

.sim-tab.active {
  color: var(--color-accent-orange);
  border-bottom-color: var(--color-accent-orange);
  font-weight: 600;
}

.tab-icon { font-size: 1rem; }

/* Layout */
.sim-content { animation: fadeIn 0.2s ease; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

.sim-grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
}

@media (max-width: 1024px) {
  .sim-grid-2 { grid-template-columns: 1fr; }
}

/* Card */
.card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0.5rem;
  padding: 1.25rem;
}

.card-title {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: var(--color-text-primary);
}

/* Form Elements */
.label-xs {
  display: block;
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--color-text-secondary);
  margin-bottom: 2px;
}

.input-sm {
  display: block;
  width: 100%;
  padding: 6px 8px;
  border-radius: 4px;
  font-size: 0.8125rem;
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
}
.input-sm:focus {
  outline: none;
  border-color: var(--color-accent-orange);
}

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }

.divider {
  height: 1px;
  background-color: var(--color-border);
  margin: 12px 0;
}

.subsection-title {
  font-size: 0.8125rem;
  font-weight: 600;
  color: var(--color-accent-orange);
  margin-bottom: 8px;
}

/* Buttons */
.btn-primary {
  width: 100%;
  padding: 8px 16px;
  border-radius: 6px;
  font-weight: 600;
  font-size: 0.875rem;
  background-color: var(--color-accent-orange);
  color: #fff;
  border: none;
  cursor: pointer;
  transition: opacity 0.2s;
}
.btn-primary:hover:not(:disabled) { opacity: 0.9; }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

.btn-outline-sm {
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 500;
  background: none;
  border: 1px dashed var(--color-border);
  color: var(--color-text-secondary);
  cursor: pointer;
}
.btn-outline-sm:hover {
  border-color: var(--color-accent-orange);
  color: var(--color-accent-orange);
}

.btn-icon-sm {
  background: none;
  border: none;
  color: var(--color-text-secondary);
  cursor: pointer;
  font-size: 0.75rem;
  padding: 2px 4px;
}
.btn-icon-sm:hover { color: #ef4444; }

/* Component Blocks */
.component-block {
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 10px;
  margin-bottom: 8px;
}

.comp-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.comp-badge {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 0.75rem;
  font-weight: 700;
}

.comp-name {
  flex: 1;
  font-weight: 600;
  font-size: 0.8125rem;
}

.comp-fields { display: flex; flex-direction: column; gap: 6px; }

.add-comp { display: flex; gap: 8px; margin-top: 8px; }

/* Metrics */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}

.metric-card {
  padding: 10px;
  border-radius: 6px;
  background-color: var(--color-bg-card);
  border: 1px solid var(--color-border);
}

.metric-card.accent {
  border-color: var(--color-accent-orange);
  background-color: rgba(234, 88, 12, 0.05);
}

.metric-label {
  font-size: 0.6875rem;
  color: var(--color-text-secondary);
  margin-bottom: 2px;
}

.metric-value {
  font-size: 1rem;
  font-weight: 700;
  color: var(--color-text-primary);
}

/* Gain Chain */
.chain-flow {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.chain-node {
  text-align: center;
  padding: 6px 12px;
  border-radius: 6px;
  background-color: var(--color-bg-card);
  border: 1px solid var(--color-border);
  min-width: 60px;
}

.chain-label {
  font-size: 0.625rem;
  color: var(--color-text-secondary);
  text-transform: capitalize;
}

.chain-val {
  font-size: 0.875rem;
  font-weight: 700;
  color: var(--color-accent-orange);
}

.chain-arrow {
  font-size: 1.25rem;
  color: var(--color-text-secondary);
}

/* Data Table */
.data-table {
  width: 100%;
  font-size: 0.8125rem;
  border-collapse: collapse;
}

.data-table th {
  text-align: left;
  padding: 4px 8px;
  font-weight: 600;
  font-size: 0.75rem;
  color: var(--color-text-secondary);
  border-bottom: 1px solid var(--color-border);
}

.data-table td {
  padding: 4px 8px;
  border-bottom: 1px solid rgba(128, 128, 128, 0.1);
}

.data-table .row-highlight {
  background-color: rgba(234, 88, 12, 0.1);
  font-weight: 700;
}

.mode-tag {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 10px;
  font-size: 0.6875rem;
  color: #fff;
}

.severity-tag {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 10px;
  font-size: 0.6875rem;
  color: #fff;
}

/* Upload */
.upload-zone {
  border: 1px dashed var(--color-border);
  border-radius: 6px;
  padding: 12px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s;
}
.upload-zone:hover {
  border-color: var(--color-accent-orange);
}

.upload-hint {
  font-size: 0.8125rem;
  color: var(--color-text-secondary);
}

.upload-info {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font-size: 0.8125rem;
  color: var(--color-text-primary);
}

.file-icon { font-size: 1.25rem; }

.cad-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 6px;
  font-size: 0.75rem;
  color: #22c55e;
}

.badge-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background-color: #22c55e;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.hidden { display: none; }

/* Uniformity Bar */
.uniformity-bar {
  width: 100%;
  height: 8px;
  background-color: var(--color-bg-card);
  border-radius: 4px;
  overflow: hidden;
}

.uniformity-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s ease;
}

.uniformity-label {
  font-size: 0.75rem;
  font-weight: 500;
  margin-top: 4px;
}

/* KV Items */
.kv-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  border-bottom: 1px solid rgba(128, 128, 128, 0.1);
}

.kv-key {
  color: var(--color-text-secondary);
  font-size: 0.75rem;
}

.kv-val {
  font-size: 0.8125rem;
  font-weight: 600;
}

/* Misc */
.solve-info {
  margin-top: 12px;
  padding-top: 8px;
  border-top: 1px solid var(--color-border);
  font-size: 0.6875rem;
  color: var(--color-text-secondary);
}

.error-box {
  padding: 12px;
  border-radius: 6px;
  background-color: rgba(220, 38, 38, 0.1);
  color: #dc2626;
  font-size: 0.8125rem;
}

.placeholder-text {
  text-align: center;
  padding: 3rem 1rem;
  color: var(--color-text-secondary);
  font-size: 0.875rem;
}

.recipe-id {
  font-size: 0.75rem;
  color: var(--color-text-secondary);
}

.rec-list {
  list-style: disc;
  padding-left: 1.25rem;
  font-size: 0.8125rem;
  color: var(--color-text-primary);
  line-height: 1.6;
}

.validation-badge {
  display: inline-block;
  padding: 4px 16px;
  border-radius: 4px;
  font-size: 0.8125rem;
  font-weight: 700;
}
.validation-badge.pass {
  background-color: rgba(34, 197, 94, 0.1);
  color: #22c55e;
}
.validation-badge.fail {
  background-color: rgba(220, 38, 38, 0.1);
  color: #dc2626;
}

.text-right { text-align: right; }
.text-center { text-align: center; }
.font-mono { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; }
.mt-3 { margin-top: 0.75rem; }
.mt-4 { margin-top: 1rem; }
.space-y-3 > :not(:first-child) { margin-top: 0.75rem; }
.max-h-48 { max-height: 12rem; }
.max-h-40 { max-height: 10rem; }
.overflow-y-auto { overflow-y: auto; }
</style>
