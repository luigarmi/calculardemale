const state = {
    featureOrder: [],
    featureSchema: null,
    models: [],
    classLabels: [],
    classMetadata: [],
    dataset: null,
    targetColumn: null,
};

const selectors = {
    individualForm: () => document.getElementById('individual-form'),
    individualModel: () => document.getElementById('individual-model'),
    featureInputs: () => document.getElementById('feature-inputs'),
    individualStatus: () => document.getElementById('individual-status'),
    individualResult: () => document.getElementById('individual-result'),
    batchForm: () => document.getElementById('batch-form'),
    batchModel: () => document.getElementById('batch-model'),
    batchStatus: () => document.getElementById('batch-status'),
    batchResult: () => document.getElementById('batch-result'),
    batchMetrics: () => document.getElementById('batch-metrics'),
    confusionImage: () => document.getElementById('confusion-image'),
    confusionTable: () => document.getElementById('confusion-table'),
    batchPreview: () => document.getElementById('batch-preview'),
    referenceContent: () => document.getElementById('reference-content'),
    classLegend: () => document.getElementById('class-legend'),
    heroClassCount: () => document.getElementById('hero-class-count'),
    heroFeatureCount: () => document.getElementById('hero-feature-count'),
};

function setStatus(element, message, type = '') {
    const el = typeof element === 'function' ? element() : element;
    if (!el) return;
    el.textContent = message;
    el.classList.remove('error', 'success');
    if (type) {
        el.classList.add(type);
    }
}

function toggleHidden(element, hidden) {
    const el = typeof element === 'function' ? element() : element;
    if (!el) return;
    if (hidden) {
        el.classList.add('hidden');
    } else {
        el.classList.remove('hidden');
    }
}

async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) {
            throw new Error('No se pudo obtener la configuracion inicial.');
        }
        const payload = await response.json();
        state.featureOrder = payload.feature_order || [];
        state.featureSchema = payload.feature_schema || null;
        state.models = payload.models || [];
        state.classLabels = payload.class_labels || [];
        state.classMetadata = payload.class_metadata || [];
        state.dataset = payload.dataset || null;
        state.targetColumn = payload.dataset?.target_column || null;

        if ((!state.classLabels || state.classLabels.length === 0) && Array.isArray(state.classMetadata)) {
            state.classLabels = state.classMetadata.map((item) => item.id);
        }

        populateModelOptions(selectors.individualModel());
        populateModelOptions(selectors.batchModel());
        renderFeatureInputs();
        renderReferenceMetrics();
        renderClassLegend();
        updateHeroStats();
        setStatus(selectors.individualStatus, 'Listo para predecir.', 'success');
        setStatus(selectors.batchStatus, 'Listo para procesar.', 'success');
    } catch (error) {
        console.error(error);
        setStatus(selectors.individualStatus, 'Error al cargar configuracion.', 'error');
        setStatus(selectors.batchStatus, 'Error al cargar configuracion.', 'error');
    }
}

function populateModelOptions(selectElement) {
    if (!selectElement) return;
    selectElement.innerHTML = '';
    state.models.forEach((model) => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.label;
        selectElement.appendChild(option);
    });
}

function renderFeatureInputs() {
    const container = selectors.featureInputs();
    if (!container) return;
    container.innerHTML = '';
    const schema = state.featureSchema;
    if (!schema || !Array.isArray(schema.sections) || schema.sections.length === 0) {
        state.featureOrder.forEach((feature) => {
            const inputId = featureInputId(feature);
            const wrapper = document.createElement('div');
            wrapper.className = 'form-row';

            const label = document.createElement('label');
            label.setAttribute('for', inputId);
            label.textContent = feature;

            const input = document.createElement('input');
            input.type = 'number';
            input.step = 'any';
            input.required = true;
            input.id = inputId;
            input.name = feature;
            input.dataset.feature = feature;
            input.placeholder = 'Valor numerico';

            wrapper.append(label, input);
            container.appendChild(wrapper);
        });
        return;
    }

    const fragment = document.createDocumentFragment();
    schema.sections.forEach((section) => {
        const sectionEl = document.createElement('section');
        sectionEl.className = 'feature-section';

        const header = document.createElement('header');
        header.className = 'feature-section-header';
        const title = document.createElement('h3');
        title.textContent = section.name;
        header.appendChild(title);
        if (section.description) {
            const subtitle = document.createElement('p');
            subtitle.textContent = section.description;
            header.appendChild(subtitle);
        }
        sectionEl.appendChild(header);

        const grid = document.createElement('div');
        grid.className = 'feature-grid';

        (section.items || []).forEach((feature) => {
            const field = buildFeatureField(feature);
            if (field) grid.appendChild(field);
        });

        sectionEl.appendChild(grid);
        fragment.appendChild(sectionEl);
    });

    container.appendChild(fragment);
}

function buildFeatureField(feature) {
    if (!feature || !feature.id) return null;
    const wrapper = document.createElement('div');
    wrapper.className = 'feature-field';

    const label = document.createElement('label');
    const inputId = featureInputId(feature.id);
    label.setAttribute('for', inputId);
    label.className = 'field-label';
    label.textContent = feature.label || feature.id;

    const control = createFeatureControl(feature, inputId);
    if (!control) return null;

    const description = document.createElement('small');
    description.className = 'field-hint';
    description.textContent = buildHelperText(feature);

    wrapper.append(label, control, description);
    return wrapper;
}

function getFeatureMeta(featureId) {
    if (!state.featureSchema || !state.featureSchema.by_id) return null;
    return state.featureSchema.by_id[featureId] || null;
}

function attachFieldListeners(element) {
    const removeError = () => {
        element.classList.remove('input-error');
    };
    element.addEventListener('input', removeError);
    element.addEventListener('change', removeError);
}

function raiseInputError(element, message) {
    if (element) {
        element.classList.add('input-error');
        if (typeof element.focus === 'function') {
            element.focus();
        }
    }
    throw new Error(message);
}

function resolveClassMeta(classId) {
    const numericId = Number(classId);
    if (!Number.isFinite(numericId)) return null;
    const metadata = Array.isArray(state.classMetadata) ? state.classMetadata : [];
    return metadata.find((item) => Number(item.id) === numericId) || null;
}

function resolveClassLabel(classId) {
    const meta = resolveClassMeta(classId);
    if (meta && meta.label) return meta.label;
    return `Clase ${classId}`;
}

function createFeatureControl(feature, inputId) {
    const bounds = deriveBounds(feature);
    const defaultValue = determineDefaultValue(feature, bounds);
    if (feature.type === 'binary') {
        const select = document.createElement('select');
        select.id = inputId;
        select.name = feature.id;
        select.dataset.feature = feature.id;
        select.required = true;
        select.dataset.min = bounds.min ?? '';
        select.dataset.max = bounds.max ?? '';

        const placeholderOption = document.createElement('option');
        placeholderOption.value = '';
        placeholderOption.textContent = 'Selecciona una opcion';
        placeholderOption.disabled = true;
        placeholderOption.selected = defaultValue === null || defaultValue === undefined;
        select.appendChild(placeholderOption);

        const options = [
            { value: 0, label: 'No (0)' },
            { value: 1, label: 'Si (1)' },
        ];
        options.forEach((option) => {
            const opt = document.createElement('option');
            opt.value = option.value;
            opt.textContent = option.label;
            select.appendChild(opt);
        });
        if (defaultValue !== null && defaultValue !== undefined) {
            select.value = String(defaultValue);
        }
        attachFieldListeners(select);
        return select;
    }

    const input = document.createElement('input');
    input.type = 'number';
    input.id = inputId;
    input.name = feature.id;
    input.dataset.feature = feature.id;
    input.required = true;
    input.step = feature.step || 'any';
    input.placeholder = buildPlaceholder(feature, bounds);
    if (bounds.min !== null && bounds.min !== undefined) {
        input.min = bounds.min;
        input.dataset.min = bounds.min;
    }
    if (bounds.max !== null && bounds.max !== undefined) {
        input.max = bounds.max;
        input.dataset.max = bounds.max;
    }
    if (defaultValue !== null && defaultValue !== undefined) {
        input.value = defaultValue;
    }
    attachFieldListeners(input);
    return input;
}

function determineDefaultValue(feature, bounds) {
    if (feature.type === 'binary') {
        const mean = Number(feature.mean);
        if (!Number.isNaN(mean)) {
            return mean >= 0.5 ? 1 : 0;
        }
        return 0;
    }
    return null;
}

function buildPlaceholder(feature, bounds) {
    const mean = Number(feature.mean);
    if (feature.type === 'binary') {
        return '0 = No, 1 = Si';
    }
    if (!Number.isNaN(mean)) {
        return `Ej: ${formatNumber(mean, feature.type)}`;
    }
    if (bounds.min !== null && bounds.min !== undefined && bounds.max !== null && bounds.max !== undefined) {
        return `Rango ${formatNumber(bounds.min, feature.type)} - ${formatNumber(bounds.max, feature.type)}`;
    }
    return 'Ingresa un valor';
}

function deriveBounds(feature) {
    const allowed = Array.isArray(feature.allowed_values) ? feature.allowed_values : null;
    if (allowed && allowed.length) {
        const minAllowed = Math.min(...allowed);
        const maxAllowed = Math.max(...allowed);
        return { min: minAllowed, max: maxAllowed };
    }
    const minRaw = typeof feature.min === 'number' ? feature.min : null;
    const maxRaw = typeof feature.max === 'number' ? feature.max : null;
    const strict = Boolean(feature.strict_bounds);
    if (minRaw === null && maxRaw === null) {
        if (feature.type === 'integer' || feature.type === 'number') {
            return { min: 0, max: null };
        }
        return { min: null, max: null };
    }
    if (strict) {
        const minBound = minRaw !== null ? Math.max(0, minRaw) : null;
        return { min: minBound, max: maxRaw };
    }
    if (minRaw !== null && maxRaw !== null) {
        const span = maxRaw - minRaw;
        const padding = span ? span * 0.1 : Math.max(Math.abs(maxRaw) * 0.1, 0.1);
        const minBound = Math.max(0, minRaw - padding);
        const maxBound = maxRaw + padding;
        return { min: minBound, max: maxBound };
    }
    return {
        min: minRaw !== null ? Math.max(0, minRaw) : null,
        max: maxRaw,
    };
}

function roundValue(value, featureType) {
    if (featureType === 'integer') {
        return Math.round(value);
    }
    if (featureType === 'binary') {
        return value >= 0.5 ? 1 : 0;
    }
    return Number(value.toFixed(2));
}

function buildHelperText(feature) {
    const parts = [];
    if (feature.description) {
        parts.push(feature.description);
    }
    const bounds = deriveBounds(feature);
    if (bounds.min !== null && bounds.min !== undefined && bounds.max !== null && bounds.max !== undefined) {
        parts.push(`Rango sugerido: ${formatNumber(bounds.min, feature.type)} - ${formatNumber(bounds.max, feature.type)}.`);
    } else if (bounds.min !== null && bounds.min !== undefined) {
        parts.push(`Minimo: ${formatNumber(bounds.min, feature.type)}.`);
    }
    if (feature.unit) {
        parts.push(`Unidad: ${feature.unit}.`);
    }
    if (feature.type === 'binary') {
        parts.push('Utiliza 0 para No y 1 para Si.');
    }
    return parts.join(' ');
}

function formatNumber(value, featureType) {
    if (featureType === 'binary' || featureType === 'integer') {
        return String(Math.round(value));
    }
    if (value >= 1000) {
        return Number.parseFloat(value).toFixed(0);
    }
    return Number.parseFloat(value).toFixed(2);
}

function renderReferenceMetrics() {
    const container = selectors.referenceContent();
    if (!container) return;
    container.innerHTML = '';
    const fragment = document.createDocumentFragment();

    if (state.dataset) {
        const datasetCard = document.createElement('div');
        datasetCard.className = 'metric-card info-card';
        const title = document.createElement('h4');
        title.textContent = 'Dataset activo';
        const description = document.createElement('strong');
        description.textContent = state.dataset.description || state.dataset.id || 'Dataset clinico';
        const detail = document.createElement('p');
        const target = state.targetColumn || 'diagnosis';
        const classNames = (state.classMetadata || []).map((item) => item.label).filter(Boolean);
        const classText = classNames.length
            ? `Clases: ${classNames.join(', ')}.`
            : 'Clases no disponibles.';
        detail.textContent = `Variable objetivo: ${target}. ${classText}`;
        datasetCard.append(title, description, detail);
        fragment.appendChild(datasetCard);
    }

    if (Array.isArray(state.models) && state.models.length) {
        state.models.forEach((model) => {
            const metricCard = document.createElement('div');
            metricCard.className = 'metric-card info-card';
            const title = document.createElement('h4');
            title.textContent = model.label || model.id;
            const detail = document.createElement('p');
            const metrics = model.metrics?.metrics || model.metrics || {};
            const parts = [];
            if (typeof metrics.accuracy === 'number') parts.push(`Acc ${formatPercent(metrics.accuracy)}`);
            if (typeof metrics.precision === 'number') parts.push(`Prec ${formatPercent(metrics.precision)}`);
            if (typeof metrics.recall === 'number') parts.push(`Rec ${formatPercent(metrics.recall)}`);
            if (typeof metrics.f1 === 'number') parts.push(`F1 ${formatPercent(metrics.f1)}`);
            detail.textContent = parts.length ? parts.join(' | ') : 'Metricas no disponibles.';
            metricCard.append(title, detail);
            fragment.appendChild(metricCard);
        });
    }

    const tipsCard = document.createElement('div');
    tipsCard.className = 'metric-card info-card';
    const tipsTitle = document.createElement('h4');
    tipsTitle.textContent = 'Recomendaciones rapidas';
    const tipsList = document.createElement('ul');
    tipsList.className = 'metric-list';
    [
        'Los campos binarios solo aceptan 0 (No) o 1 (Si).',
        'Los rangos sugeridos se basan en el dataset original.',
        'Verifica las unidades antes de digitar valores de laboratorio.',
    ].forEach((tip) => {
        const item = document.createElement('li');
        item.textContent = tip;
        tipsList.appendChild(item);
    });
    tipsCard.append(tipsTitle, tipsList);
    fragment.appendChild(tipsCard);

    container.appendChild(fragment);
}

function renderClassLegend() {
    const container = selectors.classLegend();
    if (!container) return;
    container.innerHTML = '';
    const metadata = Array.isArray(state.classMetadata) ? state.classMetadata : [];
    if (metadata.length === 0) {
        container.textContent = 'Sin informacion de clases disponible.';
        return;
    }

    const fragment = document.createDocumentFragment();
    metadata.forEach((item) => {
        const entry = document.createElement('div');
        entry.className = 'legend-entry';
        const code = document.createElement('code');
        code.textContent = `Clase ${item.id}`;
        const name = document.createElement('h3');
        name.textContent = item.label;
        const description = document.createElement('p');
        description.textContent = item.description;
        entry.append(code, name, description);
        fragment.appendChild(entry);
    });

    container.appendChild(fragment);
}

function updateHeroStats() {
    const classCountEl = selectors.heroClassCount();
    const featureCountEl = selectors.heroFeatureCount();
    if (classCountEl) {
        const classCount = Array.isArray(state.classLabels) ? state.classLabels.length : 0;
        classCountEl.textContent = classCount || '-';
        const hint = classCountEl.nextElementSibling;
        if (hint) {
            const classNames = (state.classMetadata || []).map((item) => item.label).filter(Boolean);
            hint.textContent = classNames.length
                ? `Diagnosticos: ${classNames.join(', ')}`
                : 'Diagnosticos: -';
        }
    }
    if (featureCountEl) {
        const featureCount = Array.isArray(state.featureOrder) ? state.featureOrder.length : 0;
        featureCountEl.textContent = featureCount || '-';
    }
}

async function handleIndividualSubmit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const statusEl = selectors.individualStatus();
    toggleHidden(selectors.individualResult, true);
    setStatus(statusEl, 'Calculando...', '');
    form.querySelector('button[type="submit"]').disabled = true;

    try {
        const model = selectors.individualModel().value;
        const features = {};
        const fields = selectors.featureInputs().querySelectorAll('[data-feature]');
        fields.forEach((element) => element.classList.remove('input-error'));
        fields.forEach((element) => {
            const featureId = element.dataset.feature;
            if (!featureId) return;
            const meta = getFeatureMeta(featureId);
            const label = meta?.label || featureId;
            const rawValue = element.value;
            if (rawValue === '' || rawValue === null || rawValue === undefined) {
                raiseInputError(element, `Completa el campo ${label}.`);
            }

            let numericValue = Number(rawValue);
            if (!Number.isFinite(numericValue)) {
                raiseInputError(element, `'${label}' debe ser numerico.`);
            }

            if (meta?.type === 'binary') {
                numericValue = Math.round(numericValue);
                if (![0, 1].includes(numericValue)) {
                    raiseInputError(element, `'${label}' solo acepta 0 (No) o 1 (Si).`);
                }
            } else if (meta?.type === 'integer') {
                const rounded = Math.round(numericValue);
                if (Math.abs(rounded - numericValue) > 1e-6) {
                    raiseInputError(element, `'${label}' debe ser un numero entero.`);
                }
                numericValue = rounded;
            }

            const bounds = deriveBounds(meta || {});
            if (bounds.min !== null && bounds.min !== undefined && numericValue < bounds.min) {
                raiseInputError(
                    element,
                    `'${label}' esta por debajo del minimo permitido (${formatNumber(bounds.min, meta?.type || 'number')}).`
                );
            }
            if (bounds.max !== null && bounds.max !== undefined && numericValue > bounds.max) {
                raiseInputError(
                    element,
                    `'${label}' supera el maximo permitido (${formatNumber(bounds.max, meta?.type || 'number')}).`
                );
            }

            features[featureId] = numericValue;
        });

        const response = await fetch('/api/predict/individual', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ model, features }),
        });

        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || 'No se pudo obtener la prediccion.');
        }

        renderIndividualResult(payload);
        setStatus(statusEl, 'Prediccion generada.', 'success');
    } catch (error) {
        console.error(error);
        setStatus(selectors.individualStatus, error.message, 'error');
    } finally {
        form.querySelector('button[type="submit"]').disabled = false;
    }
}

function renderIndividualResult(payload) {
    const container = selectors.individualResult();
    if (!container) return;
    container.innerHTML = '';

    const title = document.createElement('h3');
    title.textContent = `Resultado con ${payload.model?.label || payload.model?.id}`;

    const summary = document.createElement('div');
    summary.className = 'prediction-summary';
    const predictedLabel = payload.prediction_detail?.label || resolveClassLabel(payload.prediction);
    const predictedDescription = payload.prediction_detail?.description
        || resolveClassMeta(payload.prediction)?.description
        || 'Sin descripcion disponible para esta clase.';
    const heading = document.createElement('h4');
    heading.textContent = predictedLabel;
    const description = document.createElement('p');
    description.textContent = predictedDescription;
    summary.append(heading, description);

    container.append(title, summary);

    if (Array.isArray(payload.probabilities) && payload.probabilities.length > 0) {
        const list = document.createElement('ul');
        list.className = 'probability-list';
        payload.probabilities.forEach((prob, index) => {
            const item = document.createElement('li');
            const value = typeof prob === 'object' ? prob.probability : prob;
            const classId = typeof prob === 'object'
                ? prob.id
                : state.classLabels[index] ?? index;
            const label = typeof prob === 'object' && prob.label
                ? prob.label
                : resolveClassLabel(classId);
            item.innerHTML = `<span class="prob-label">${label}</span><span class="prob-value">${formatPercent(value)}</span>`;
            const detailText = typeof prob === 'object' && prob.description
                ? prob.description
                : resolveClassMeta(classId)?.description;
            if (detailText) {
                const detail = document.createElement('small');
                detail.textContent = detailText;
                item.appendChild(detail);
            }
            list.appendChild(item);
        });
        container.appendChild(list);
    }

    toggleHidden(container, false);
}

function formatPercent(value) {
    if (typeof value !== 'number' || Number.isNaN(value)) return '0.00%';
    return `${(value * 100).toFixed(2)}%`;
}

function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn[data-tab-target]');
    if (!tabButtons.length) return;
    tabButtons.forEach((button) => {
        button.addEventListener('click', () => {
            activateTab(button.dataset.tabTarget, button);
        });
    });
    const defaultButton = Array.from(tabButtons).find((btn) => btn.classList.contains('active')) || tabButtons[0];
    if (defaultButton) {
        activateTab(defaultButton.dataset.tabTarget, defaultButton);
    }
}

function activateTab(targetId, activeButton) {
    document.querySelectorAll('.tab-btn').forEach((btn) => {
        btn.classList.toggle('active', btn === activeButton);
    });
    document.querySelectorAll('.tab-pane').forEach((pane) => {
        const isActive = pane.id === targetId;
        pane.classList.toggle('hidden', !isActive);
        pane.classList.toggle('active', isActive);
    });
}

async function handleBatchSubmit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const statusEl = selectors.batchStatus();
    setStatus(statusEl, 'Procesando archivo...', '');
    toggleHidden(selectors.batchResult, true);
    form.querySelector('button[type="submit"]').disabled = true;

    try {
        const formData = new FormData(form);
        const response = await fetch('/api/predict/batch', {
            method: 'POST',
            body: formData,
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || 'No se pudo procesar el archivo.');
        }
        renderBatchResult(payload);
        setStatus(statusEl, `Analizado ${payload.total_samples} registros.`, 'success');
    } catch (error) {
        console.error(error);
        setStatus(statusEl, error.message, 'error');
    } finally {
        form.querySelector('button[type="submit"]').disabled = false;
    }
}

function renderBatchResult(payload) {
    if (Array.isArray(payload.class_labels) && payload.class_labels.length) {
        state.classLabels = payload.class_labels;
    }
    if (Array.isArray(payload.class_metadata) && payload.class_metadata.length) {
        state.classMetadata = payload.class_metadata;
    }
    renderBatchMetrics(payload);
    renderConfusionMatrix(payload.confusion_matrix);
    renderBatchPreview(payload.preview);
    updateHeroStats();
    renderClassLegend();
    toggleHidden(selectors.batchResult, false);
}

function renderBatchMetrics(payload) {
    const container = selectors.batchMetrics();
    if (!container) return;
    container.innerHTML = '';
    const metrics = payload.metrics || {};

    const metricLabels = [
        { key: 'accuracy', label: 'Exactitud' },
        { key: 'precision', label: 'Precision' },
        { key: 'recall', label: 'Recall' },
        { key: 'f1', label: 'F1' },
    ];

    metricLabels.forEach((item) => {
        const value = metrics[item.key];
        const card = document.createElement('div');
        card.className = 'metric-card';

        const title = document.createElement('h4');
        title.textContent = item.label;

        const strong = document.createElement('strong');
        strong.textContent = typeof value === 'number' ? formatPercent(value) : 'N/A';

        card.append(title, strong);
        container.appendChild(card);
    });
}

function renderConfusionMatrix(matrixPayload) {
    const imageEl = selectors.confusionImage();
    const tableEl = selectors.confusionTable();
    if (!matrixPayload || !imageEl || !tableEl) return;

    if (matrixPayload.image_png_base64) {
        imageEl.src = `data:image/png;base64,${matrixPayload.image_png_base64}`;
    }

    const matrix = matrixPayload.matrix || [];
    const labels = matrixPayload.labels || [];

    tableEl.innerHTML = '';
    if (matrix.length === 0) return;

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    headRow.appendChild(document.createElement('th')); // empty corner
    labels.forEach((label) => {
        const th = document.createElement('th');
        th.textContent = `Pred ${resolveClassLabel(label)}`;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);

    const tbody = document.createElement('tbody');
    matrix.forEach((row, rowIndex) => {
        const tr = document.createElement('tr');
        const labelCell = document.createElement('th');
        const actualLabel = labels[rowIndex] ?? rowIndex;
        labelCell.textContent = `Real ${resolveClassLabel(actualLabel)}`;
        tr.appendChild(labelCell);
        row.forEach((value) => {
            const td = document.createElement('td');
            td.textContent = value;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });

    tableEl.appendChild(thead);
    tableEl.appendChild(tbody);
}

function renderBatchPreview(rows) {
    const table = selectors.batchPreview();
    if (!table) return;
    table.innerHTML = '';
    if (!Array.isArray(rows) || rows.length === 0) {
        table.textContent = 'Sin registros para mostrar.';
        return;
    }

    const columns = Object.keys(rows[0]);

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    columns.forEach((column) => {
        const th = document.createElement('th');
        th.textContent = column;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);

    const tbody = document.createElement('tbody');
    rows.forEach((row) => {
        const tr = document.createElement('tr');
        columns.forEach((column) => {
            const td = document.createElement('td');
            const value = row[column];
            td.textContent = typeof value === 'number'
                ? Number.parseFloat(value).toFixed(3)
                : value;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });

    table.appendChild(thead);
    table.appendChild(tbody);
}

document.addEventListener('DOMContentLoaded', () => {
    const individualForm = selectors.individualForm();
    const batchForm = selectors.batchForm();
    if (individualForm) individualForm.addEventListener('submit', handleIndividualSubmit);
    if (batchForm) batchForm.addEventListener('submit', handleBatchSubmit);
    setupTabs();
    loadStatus();
});
function featureInputId(featureName) {
    return `feature-${featureName.replace(/[^a-zA-Z0-9]/g, '_')}`;
}
