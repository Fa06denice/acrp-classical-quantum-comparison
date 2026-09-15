"""Phase 5.2.D — remote-job dedup decision (fake service only, never IBM).

Covers the deterministic tag and every decision branch/failure mode with a fake search:
zero/one/many jobs, deferred consistency, timeout, auth, quota, backend-unavailable,
malformed, job-found-after-crash, lost runs_dir/index recovered by tag, and a same-tag
incompatible-config collision. No IBM contact, no token, no network.
"""
from __future__ import annotations

import pytest

from acrpq.dashboard import qpu_dedup as D

BATCH = "sha256:" + "1" * 64
JOB = "sha256:" + "2" * 64
OTHER_JOB = "sha256:" + "3" * 64


def _ref(job_id, tag, cfg=JOB, status="DONE"):
    return D.RemoteJobRef(job_id=job_id, tags=(tag,), job_config_hash=cfg, status=status)


# --- tag ------------------------------------------------------------------
def test_tag_is_deterministic_and_provider_safe():
    t1 = D.job_tag(BATCH, JOB)
    t2 = D.job_tag(BATCH, JOB)
    assert t1 == t2 == "acrpq:" + "1" * 12 + ":" + "2" * 12
    assert len(t1) <= 63 and D._TAG_RE.match(t1)
    # no secret/path/timestamp material, only the two identity prefixes
    assert t1.count(":") == 2 and t1.startswith("acrpq:")


def test_tag_changes_with_either_identity():
    assert D.job_tag(BATCH, JOB) != D.job_tag(BATCH, OTHER_JOB)
    assert D.job_tag(BATCH, JOB) != D.job_tag(OTHER_JOB, JOB)


def test_tag_rejects_malformed_hashes():
    for bad in ["", "nope", "sha256:xyz", "1" * 64]:
        with pytest.raises(ValueError):
            D.job_tag(BATCH, bad)


# --- decision: the only path that permits a submission --------------------
def test_zero_jobs_allows_submission():
    d = D.decide_submission(batch_config_hash=BATCH, job_config_hash=JOB,
                            local_job_id=None, search=D.FakeRemoteJobService(jobs=[]))
    assert d.action == "submit_allowed" and d.may_submit is True


def test_one_matching_job_adopts_never_submits():
    tag = D.job_tag(BATCH, JOB)
    svc = D.FakeRemoteJobService(jobs=[_ref("job-1", tag)])
    d = D.decide_submission(batch_config_hash=BATCH, job_config_hash=JOB,
                            local_job_id=None, search=svc)
    assert d.action == "adopt_remote" and d.job_id == "job-1" and d.may_submit is False


def test_multiple_jobs_same_tag_fail_closed():
    tag = D.job_tag(BATCH, JOB)
    svc = D.FakeRemoteJobService(jobs=[_ref("job-1", tag), _ref("job-2", tag)])
    d = D.decide_submission(batch_config_hash=BATCH, job_config_hash=JOB,
                            local_job_id=None, search=svc)
    assert d.action == "refuse" and d.may_submit is False and "fail closed" in d.reason


def test_local_job_id_means_already_submitted():
    d = D.decide_submission(batch_config_hash=BATCH, job_config_hash=JOB,
                            local_job_id="job-local", search=D.FakeRemoteJobService(jobs=[]))
    assert d.action == "already_submitted" and d.job_id == "job-local" and d.may_submit is False


# --- every non-authoritative outcome refuses (never auto-resubmits) -------
@pytest.mark.parametrize("err", [
    D.RemoteTimeout("t"), D.RemoteAuthError("a"), D.RemoteQuotaError("q"),
    D.RemoteBackendUnavailable("b"), D.RemoteMalformedResult("m"),
    RuntimeError("unexpected"),
])
def test_search_errors_refuse_never_submit(err):
    svc = D.FakeRemoteJobService(raise_error=err) if isinstance(err, D.RemoteSearchError) \
        else _RaisingSearch(err)
    d = D.decide_submission(batch_config_hash=BATCH, job_config_hash=JOB,
                            local_job_id=None, search=svc)
    assert d.action == "refuse" and d.may_submit is False


class _RaisingSearch:
    def __init__(self, exc):
        self._exc = exc

    def find_by_tag(self, tag):
        raise self._exc


def test_malformed_result_refuses():
    d = D.decide_submission(batch_config_hash=BATCH, job_config_hash=JOB,
                            local_job_id=None, search=D.FakeRemoteJobService(malformed=True))
    assert d.action == "refuse" and "malformed" in d.reason


# --- crash / lost-state recovery by tag -----------------------------------
def test_job_found_after_local_crash_is_adopted():
    """Local job_id lost (crash) but the remote job carries the tag -> adopt, no submit."""
    tag = D.job_tag(BATCH, JOB)
    svc = D.FakeRemoteJobService(jobs=[_ref("job-remote", tag)])
    d = D.decide_submission(batch_config_hash=BATCH, job_config_hash=JOB,
                            local_job_id=None, search=svc)
    assert d.action == "adopt_remote" and d.job_id == "job-remote"


def test_lost_runs_dir_or_index_recovered_by_tag():
    """Whether the runs_dir or the batch index was lost, with local_job_id=None the tag
    search still finds and adopts the remote job — no resubmission."""
    tag = D.job_tag(BATCH, JOB)
    svc = D.FakeRemoteJobService(jobs=[_ref("job-remote", tag)])
    for _lost in ("runs_dir", "index"):
        d = D.decide_submission(batch_config_hash=BATCH, job_config_hash=JOB,
                                local_job_id=None, search=svc)
        assert d.action == "adopt_remote" and d.job_id == "job-remote"


def test_same_tag_incompatible_config_fails_closed():
    tag = D.job_tag(BATCH, JOB)
    svc = D.FakeRemoteJobService(jobs=[_ref("job-x", tag, cfg="sha256:" + "9" * 64)])
    d = D.decide_submission(batch_config_hash=BATCH, job_config_hash=JOB,
                            local_job_id=None, search=svc)
    assert d.action == "refuse" and "different config" in d.reason


def test_deferred_consistency_duplicate_is_caught_by_the_tag():
    """If deferred remote consistency let two processes both see zero and submit, both
    jobs carry the SAME tag, so a later search finds two and fails closed."""
    tag = D.job_tag(BATCH, JOB)
    empty = D.FakeRemoteJobService(jobs=[])
    assert D.decide_submission(batch_config_hash=BATCH, job_config_hash=JOB,
                               local_job_id=None, search=empty).may_submit  # both saw zero
    later = D.FakeRemoteJobService(jobs=[_ref("job-a", tag), _ref("job-b", tag)])
    d = D.decide_submission(batch_config_hash=BATCH, job_config_hash=JOB,
                            local_job_id=None, search=later)
    assert d.action == "refuse"                       # the duplicate is detectable & refused


def test_job_without_the_tag_is_ignored():
    """A remote job that does not carry our tag must not be treated as ours."""
    svc = D.FakeRemoteJobService(jobs=[_ref("unrelated", "acrpq:" + "f" * 12 + ":" + "f" * 12)])
    d = D.decide_submission(batch_config_hash=BATCH, job_config_hash=JOB,
                            local_job_id=None, search=svc)
    assert d.action == "submit_allowed"               # none carry OUR tag
