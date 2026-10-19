(function (newDoc, oldDoc, userCtx, secObj) {
  // SlicerLNQ Chronicle validator. Runs inside CouchDB on every write.
  // See ../design/README.md for what this does and does not enforce.
  // Wrapped in parens so the linter parses this as a function expression;
  // CouchDB eval()s the body as a function regardless.

  function reject(msg) { throw ({ forbidden: msg }); }
  function unauthorized(msg) { throw ({ unauthorized: msg }); }

  var isAdmin = userCtx && userCtx.roles && userCtx.roles.indexOf('_admin') !== -1;

  // _design and _local documents are admin-only; let other CouchDB rules govern shape.
  if (newDoc._id && (newDoc._id.indexOf('_design/') === 0 || newDoc._id.indexOf('_local/') === 0)) {
    if (!isAdmin) unauthorized('only admins may write design or local documents');
    return;
  }

  // Untyped documents pass through. Typed documents follow the discipline below.
  if (!newDoc.type) return;

  var TYPE_PREFIX = {
    Cohort: 'cohort:',
    Protocol: 'protocol:',
    Project: 'project:',
    CohortResolution: 'cohortresolution:',
    Annotation: 'annotation:',
    ModelGeneration: 'modelgeneration:',
    InferenceRun: 'inferencerun:',
    Blob: 'blob:',
    TrainingJob: 'trainingjob:'
  };
  var prefix = TYPE_PREFIX[newDoc.type];
  if (!prefix) reject('unknown type: ' + newDoc.type);

  // _id must carry the right type prefix.
  if (!newDoc._id || newDoc._id.indexOf(prefix) !== 0) {
    reject("_id must start with '" + prefix + "' for type " + newDoc.type);
  }

  // Immutability: no deletes, no updates of typed documents. Corrections go in new docs.
  if (newDoc._deleted) {
    reject('typed documents are immutable; write a Correction instead of deleting');
  }
  // Blob and TrainingJob are the exceptions to immutability:
  //  - Blob.locations grows/shrinks as caches come and go
  //  - TrainingJob.status/metrics/heartbeat change as the trainer runs
  // Everything else is frozen on those two types; all other typed docs are
  // strictly immutable and corrections are new versions.
  if (oldDoc && newDoc.type !== 'Blob' && newDoc.type !== 'TrainingJob') {
    reject('typed documents are immutable; write a new version with predecessor: ' + oldDoc._id);
  }
  if (oldDoc && newDoc.type === 'Blob') {
    var frozen = ['_id', 'type', 'sha256', 'size', 'mime_type',
                  'created_at', 'created_by', 'version', 'predecessor', 'name'];
    for (var fi = 0; fi < frozen.length; fi++) {
      var k = frozen[fi];
      if (JSON.stringify(newDoc[k]) !== JSON.stringify(oldDoc[k])) {
        reject('Blob.' + k + ' is immutable; only locations may change');
      }
    }
  }
  if (oldDoc && newDoc.type === 'TrainingJob') {
    var tjFrozen = ['_id', 'type', 'created_at', 'created_by', 'version', 'predecessor',
                    'name', 'project_id', 'label', 'training_annotation_ids', 'config'];
    for (var ti = 0; ti < tjFrozen.length; ti++) {
      var tk = tjFrozen[ti];
      if (JSON.stringify(newDoc[tk]) !== JSON.stringify(oldDoc[tk])) {
        reject('TrainingJob.' + tk + ' is immutable; status/metrics/heartbeat may change');
      }
    }
  }

  // Required common fields.
  if (typeof newDoc.name !== 'string' || newDoc.name.length === 0) {
    reject('name (non-empty string) is required');
  }
  if (typeof newDoc.created_at !== 'string' || newDoc.created_at.length === 0) {
    reject('created_at (ISO 8601 string) is required');
  }
  if (typeof newDoc.created_by !== 'string' || newDoc.created_by.length === 0) {
    reject('created_by (string) is required');
  }
  if (typeof newDoc.version !== 'number' || newDoc.version < 1 || newDoc.version % 1 !== 0) {
    reject('version must be a positive integer');
  }
  if (!('predecessor' in newDoc)) {
    reject('predecessor is required (null when version == 1, _id of previous version otherwise)');
  }
  if (newDoc.version === 1 && newDoc.predecessor !== null) {
    reject('version 1 documents must have predecessor: null');
  }
  if (newDoc.version > 1) {
    if (typeof newDoc.predecessor !== 'string' || newDoc.predecessor.indexOf(prefix) !== 0) {
      reject('version > 1 requires predecessor to be a ' + newDoc.type + ' _id');
    }
  }

  // Anti-forgery: writers can't claim to be someone else. Admins are exempt
  // (e.g., for migration scripts, replay from backup).
  if (!isAdmin && newDoc.created_by !== userCtx.name) {
    reject('created_by must equal the writer (' + userCtx.name + ')');
  }

  // ----- per-type structural rules -----

  if (newDoc.type === 'Cohort') {
    if (!Array.isArray(newDoc.sources) || newDoc.sources.length < 1) {
      reject('Cohort.sources must be a non-empty array');
    }
    var SOURCE_KINDS = ['idc-query', 'dicomweb', 'local-uri'];
    for (var i = 0; i < newDoc.sources.length; i++) {
      var s = newDoc.sources[i];
      if (!s || typeof s !== 'object') reject('Cohort.sources[' + i + '] must be an object');
      if (SOURCE_KINDS.indexOf(s.kind) === -1) {
        reject('Cohort.sources[' + i + '].kind must be one of ' + SOURCE_KINDS.join('|'));
      }
      if (typeof s.value !== 'string' || s.value.length === 0) {
        reject('Cohort.sources[' + i + '].value must be a non-empty string');
      }
    }
    if (newDoc.consort !== undefined) {
      var c = newDoc.consort;
      if (!c || typeof c !== 'object') reject('Cohort.consort must be an object');
      if (c.excluded !== undefined) {
        if (!Array.isArray(c.excluded)) reject('Cohort.consort.excluded must be an array');
        for (var ex = 0; ex < c.excluded.length; ex++) {
          var e = c.excluded[ex];
          if (typeof e.reason !== 'string' || typeof e.count !== 'number' || e.count < 0) {
            reject('Cohort.consort.excluded[' + ex + '] must be {reason: string, count: integer >= 0}');
          }
        }
      }
    }
  }

  else if (newDoc.type === 'Protocol') {
    if (!Array.isArray(newDoc.color_table) || newDoc.color_table.length < 1) {
      reject('Protocol.color_table must be a non-empty array');
    }
    var seenLabels = {};
    for (var j = 0; j < newDoc.color_table.length; j++) {
      var ct = newDoc.color_table[j];
      if (!ct || typeof ct !== 'object') reject('Protocol.color_table[' + j + '] must be an object');
      if (typeof ct.label !== 'number' || ct.label < 1 || ct.label % 1 !== 0) {
        reject('Protocol.color_table[' + j + '].label must be a positive integer');
      }
      if (seenLabels[ct.label]) {
        reject('Protocol.color_table label ' + ct.label + ' is duplicated');
      }
      seenLabels[ct.label] = true;
      if (typeof ct.name !== 'string' || ct.name.length === 0) {
        reject('Protocol.color_table[' + j + '].name must be a non-empty string');
      }
      if (!Array.isArray(ct.color) || ct.color.length !== 3) {
        reject('Protocol.color_table[' + j + '].color must be [r, g, b]');
      }
      for (var k = 0; k < 3; k++) {
        if (typeof ct.color[k] !== 'number' || ct.color[k] < 0 || ct.color[k] > 255 || ct.color[k] % 1 !== 0) {
          reject('Protocol.color_table[' + j + '].color[' + k + '] must be an integer 0-255');
        }
      }
      if (!ct.code || typeof ct.code !== 'object') {
        reject('Protocol.color_table[' + j + '].code must be a terminology code');
      }
      if (typeof ct.code.scheme !== 'string' || ct.code.scheme.length === 0 ||
          typeof ct.code.value !== 'string' || ct.code.value.length === 0 ||
          typeof ct.code.meaning !== 'string' || ct.code.meaning.length === 0) {
        reject('Protocol.color_table[' + j + '].code requires {scheme, value, meaning} non-empty strings');
      }
    }
  }

  else if (newDoc.type === 'TrainingJob') {
    if (typeof newDoc.project_id !== 'string' || newDoc.project_id.indexOf('project:') !== 0) {
      reject('TrainingJob.project_id must reference a Project');
    }
    if (typeof newDoc.label !== 'string' || newDoc.label.length === 0) {
      reject('TrainingJob.label must be a non-empty string');
    }
    if (!Array.isArray(newDoc.training_annotation_ids) || newDoc.training_annotation_ids.length < 1) {
      reject('TrainingJob.training_annotation_ids must be a non-empty array');
    }
    for (var tai = 0; tai < newDoc.training_annotation_ids.length; tai++) {
      if (typeof newDoc.training_annotation_ids[tai] !== 'string' ||
          newDoc.training_annotation_ids[tai].indexOf('annotation:') !== 0) {
        reject('TrainingJob.training_annotation_ids[' + tai + '] must be annotation:...');
      }
    }
    if (!newDoc.config || typeof newDoc.config !== 'object') {
      reject('TrainingJob.config must be an object');
    }
    if (typeof newDoc.config.framework !== 'string' || newDoc.config.framework.length === 0) {
      reject('TrainingJob.config.framework must be a non-empty string');
    }
    if (typeof newDoc.config.config_name !== 'string' || newDoc.config.config_name.length === 0) {
      reject('TrainingJob.config.config_name must be a non-empty string');
    }
    var STATUSES = ['pending', 'running', 'converged', 'failed', 'cancelled'];
    if (STATUSES.indexOf(newDoc.status) === -1) {
      reject('TrainingJob.status must be ' + STATUSES.join('|'));
    }
    if (newDoc.model_generation_id !== undefined && newDoc.model_generation_id !== null) {
      if (typeof newDoc.model_generation_id !== 'string' ||
          newDoc.model_generation_id.indexOf('modelgeneration:') !== 0) {
        reject('TrainingJob.model_generation_id must reference a ModelGeneration');
      }
    }
  }

  else if (newDoc.type === 'Blob') {
    // _id must be 'blob:' + 64 lowercase hex chars; sha256 must match.
    var idHash = newDoc._id.substring('blob:'.length);
    if (!/^[0-9a-f]{64}$/.test(idHash)) {
      reject('Blob._id suffix must be 64 lowercase hex chars (sha256)');
    }
    if (newDoc.sha256 !== idHash) {
      reject('Blob.sha256 must equal the suffix of _id');
    }
    if (typeof newDoc.size !== 'number' || newDoc.size < 1 || newDoc.size % 1 !== 0) {
      reject('Blob.size must be a positive integer');
    }
    if (!Array.isArray(newDoc.locations)) {
      reject('Blob.locations must be an array');
    }
    for (var li = 0; li < newDoc.locations.length; li++) {
      var loc = newDoc.locations[li];
      if (!loc || typeof loc !== 'object') {
        reject('Blob.locations[' + li + '] must be an object');
      }
      if (['local-uri', 'swift', 'https', 'equivalent-volume'].indexOf(loc.kind) === -1) {
        reject('Blob.locations[' + li + '].kind must be local-uri|swift|https|equivalent-volume');
      }
      if (typeof loc.value !== 'string' || loc.value.length === 0) {
        reject('Blob.locations[' + li + '].value must be a non-empty string');
      }
    }
  }

  else if (newDoc.type === 'CohortResolution') {
    if (typeof newDoc.cohort_id !== 'string' || newDoc.cohort_id.indexOf('cohort:') !== 0) {
      reject('CohortResolution.cohort_id must reference a Cohort (cohort:...)');
    }
    if (typeof newDoc.resolved_at !== 'string' || newDoc.resolved_at.length === 0) {
      reject('CohortResolution.resolved_at (ISO 8601 string) is required');
    }
    if (!Array.isArray(newDoc.cases)) {
      reject('CohortResolution.cases must be an array');
    }
    for (var ci = 0; ci < newDoc.cases.length; ci++) {
      var cs = newDoc.cases[ci];
      if (!cs || typeof cs !== 'object') {
        reject('CohortResolution.cases[' + ci + '] must be an object');
      }
      if (typeof cs.case_id !== 'string' || cs.case_id.length === 0) {
        reject('CohortResolution.cases[' + ci + '].case_id must be a non-empty string');
      }
      if (cs.ct_ref !== undefined && cs.ct_ref !== null) {
        if (typeof cs.ct_ref !== 'object') {
          reject('CohortResolution.cases[' + ci + '].ct_ref must be an object or null');
        }
        // Two accepted shapes: blob-keyed {blob_id} or legacy {kind, value}.
        if (cs.ct_ref.blob_id) {
          if (typeof cs.ct_ref.blob_id !== 'string' || cs.ct_ref.blob_id.indexOf('blob:') !== 0) {
            reject('CohortResolution.cases[' + ci + '].ct_ref.blob_id must reference a Blob (blob:...)');
          }
        } else {
          if (['local-uri', 'dicomweb'].indexOf(cs.ct_ref.kind) === -1) {
            reject('CohortResolution.cases[' + ci + '].ct_ref.kind must be local-uri|dicomweb (or use blob_id)');
          }
          if (typeof cs.ct_ref.value !== 'string' || cs.ct_ref.value.length === 0) {
            reject('CohortResolution.cases[' + ci + '].ct_ref.value must be a non-empty string');
          }
        }
      }
    }
  }

  else if (newDoc.type === 'Annotation') {
    if (typeof newDoc.project_id !== 'string' || newDoc.project_id.indexOf('project:') !== 0) {
      reject('Annotation.project_id must reference a Project (project:...)');
    }
    if (typeof newDoc.case_id !== 'string' || newDoc.case_id.length === 0) {
      reject('Annotation.case_id must be a non-empty string');
    }
    var STATUSES = ['todo', 'in_progress', 'submitted_for_review', 'needs_changes', 'approved', 'needs_consultation'];
    if (STATUSES.indexOf(newDoc.status) === -1) {
      reject('Annotation.status must be one of ' + STATUSES.join('|'));
    }
    if (!newDoc.producer || typeof newDoc.producer !== 'object') {
      reject('Annotation.producer must be an object');
    }
    if (['manual', 'model', 'review'].indexOf(newDoc.producer.kind) === -1) {
      reject('Annotation.producer.kind must be manual|model|review');
    }
    if (newDoc.seg_ref !== undefined && newDoc.seg_ref !== null) {
      if (typeof newDoc.seg_ref !== 'object') {
        reject('Annotation.seg_ref must be an object or null');
      }
      // Two accepted shapes: blob-keyed {blob_id} or legacy {kind, value}.
      if (newDoc.seg_ref.blob_id) {
        if (typeof newDoc.seg_ref.blob_id !== 'string' || newDoc.seg_ref.blob_id.indexOf('blob:') !== 0) {
          reject('Annotation.seg_ref.blob_id must reference a Blob (blob:...)');
        }
      } else {
        if (['local-uri', 'dicomweb'].indexOf(newDoc.seg_ref.kind) === -1) {
          reject('Annotation.seg_ref.kind must be local-uri|dicomweb (or use blob_id)');
        }
        if (typeof newDoc.seg_ref.value !== 'string' || newDoc.seg_ref.value.length === 0) {
          reject('Annotation.seg_ref.value must be a non-empty string');
        }
      }
    }
  }

  else if (newDoc.type === 'ModelGeneration') {
    if (typeof newDoc.project_id !== 'string' || newDoc.project_id.indexOf('project:') !== 0) {
      reject('ModelGeneration.project_id must reference a Project (project:...)');
    }
    if (typeof newDoc.label !== 'string' || newDoc.label.length === 0) {
      reject('ModelGeneration.label must be a non-empty string');
    }
    if (!Array.isArray(newDoc.training_annotation_ids) || newDoc.training_annotation_ids.length < 1) {
      reject('ModelGeneration.training_annotation_ids must be a non-empty array of annotation: refs');
    }
    for (var ai = 0; ai < newDoc.training_annotation_ids.length; ai++) {
      var aid = newDoc.training_annotation_ids[ai];
      if (typeof aid !== 'string' || aid.indexOf('annotation:') !== 0) {
        reject('ModelGeneration.training_annotation_ids[' + ai + '] must be annotation:...');
      }
    }
    if (!newDoc.weights_ref || typeof newDoc.weights_ref !== 'object') {
      reject('ModelGeneration.weights_ref must be an object');
    }
    // Two accepted shapes: blob-keyed {blob_id} or legacy {kind, value}.
    if (newDoc.weights_ref.blob_id) {
      if (typeof newDoc.weights_ref.blob_id !== 'string' || newDoc.weights_ref.blob_id.indexOf('blob:') !== 0) {
        reject('ModelGeneration.weights_ref.blob_id must reference a Blob (blob:...)');
      }
    } else {
      if (['local-uri', 'swift', 'https'].indexOf(newDoc.weights_ref.kind) === -1) {
        reject('ModelGeneration.weights_ref.kind must be local-uri|swift|https (or use blob_id)');
      }
      if (typeof newDoc.weights_ref.value !== 'string' || newDoc.weights_ref.value.length === 0) {
        reject('ModelGeneration.weights_ref.value must be a non-empty string');
      }
    }
  }

  else if (newDoc.type === 'InferenceRun') {
    if (typeof newDoc.project_id !== 'string' || newDoc.project_id.indexOf('project:') !== 0) {
      reject('InferenceRun.project_id must reference a Project (project:...)');
    }
    if (typeof newDoc.model_generation_id !== 'string' || newDoc.model_generation_id.indexOf('modelgeneration:') !== 0) {
      reject('InferenceRun.model_generation_id must reference a ModelGeneration (modelgeneration:...)');
    }
    if (!Array.isArray(newDoc.case_ids) || newDoc.case_ids.length < 1) {
      reject('InferenceRun.case_ids must be a non-empty array');
    }
    for (var ki = 0; ki < newDoc.case_ids.length; ki++) {
      if (typeof newDoc.case_ids[ki] !== 'string' || newDoc.case_ids[ki].length === 0) {
        reject('InferenceRun.case_ids[' + ki + '] must be a non-empty string');
      }
    }
    if (newDoc.output_annotation_ids !== undefined && newDoc.output_annotation_ids !== null) {
      if (!Array.isArray(newDoc.output_annotation_ids)) {
        reject('InferenceRun.output_annotation_ids must be an array or null');
      }
      for (var oi = 0; oi < newDoc.output_annotation_ids.length; oi++) {
        var oid = newDoc.output_annotation_ids[oi];
        if (typeof oid !== 'string' || oid.indexOf('annotation:') !== 0) {
          reject('InferenceRun.output_annotation_ids[' + oi + '] must be annotation:...');
        }
      }
    }
  }

  else if (newDoc.type === 'Project') {
    if (typeof newDoc.cohort_id !== 'string' || newDoc.cohort_id.indexOf('cohort:') !== 0) {
      reject('Project.cohort_id must reference a Cohort (cohort:...)');
    }
    if (typeof newDoc.protocol_id !== 'string' || newDoc.protocol_id.indexOf('protocol:') !== 0) {
      reject('Project.protocol_id must reference a Protocol (protocol:...)');
    }
    if (!Array.isArray(newDoc.members) || newDoc.members.length < 1) {
      reject('Project.members must be a non-empty array');
    }
    var ROLES = ['admin', 'annotator', 'reviewer'];
    var hasAdmin = false;
    for (var m = 0; m < newDoc.members.length; m++) {
      var member = newDoc.members[m];
      if (!member || typeof member !== 'object') reject('Project.members[' + m + '] must be an object');
      if (typeof member.user !== 'string' || member.user.length === 0) {
        reject('Project.members[' + m + '].user must be a non-empty string');
      }
      if (ROLES.indexOf(member.role) === -1) {
        reject('Project.members[' + m + '].role must be one of ' + ROLES.join('|'));
      }
      if (member.role === 'admin') hasAdmin = true;
    }
    if (!hasAdmin) reject('Project.members must include at least one admin');
  }
});
