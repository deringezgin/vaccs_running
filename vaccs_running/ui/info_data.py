from __future__ import annotations

import threading

from ..slurm import (
    JOB_EFFICIENCY_WINDOWS,
    USER_INFO_WINDOWS,
)


class InfoDataMixin:
    def _ensure_info_loaded(self) -> None:
        if not self._info_started:
            self._start_info_refresh()

    def _info_loading(self) -> bool:
        with self._info_lock:
            status = self._info_data.get("status")
            if status == "loading":
                return True
            if status == "ready" and self._info_data.get("forecast") is None:
                return True
            efficiency = self._info_data.get("efficiency")
            return isinstance(efficiency, dict) and any(
                value is None for value in efficiency.values()
            )

    def _start_info_refresh(self) -> bool:
        """Kick off the info fetch.

        The account/usage/storage card loads in one thread; each job-efficiency
        window (7d/30d/1y) loads in its own thread so it pops in as it returns
        (the 1-year sacct is the slow one). All threads write into
        ``_info_data`` under the lock, guarded by a generation counter.
        """
        if self._info_loading():
            return False
        self._info_started = True
        with self._info_lock:
            self._info_generation += 1
            generation = self._info_generation
            self._info_data = {
                "status": "loading",
                "fairshare": {},
                "forecast": None,
                "forecast_error": "",
                "default": "",
                "accounts_error": "",
                "windows": {},
                "gpfs": None,
                "gpfs_error": "",
                "gpfs_group_usage": None,
                "gpfs_group_usage_error": "",
                # None per window == still loading.
                "efficiency": {key: None for key, _w, _l in JOB_EFFICIENCY_WINDOWS},
            }
        threading.Thread(
            target=self._fetch_info_base, args=(generation,), daemon=True
        ).start()
        threading.Thread(
            target=self._fetch_info_forecast, args=(generation,), daemon=True
        ).start()
        for key, window, label in JOB_EFFICIENCY_WINDOWS:
            threading.Thread(
                target=self._fetch_info_efficiency,
                args=(generation, key, window, label),
                daemon=True,
            ).start()
        return True

    def _fetch_info_base(self, generation: int) -> None:
        """Load accounts, compute usage, and storage (the always-shown card)."""
        update: dict[str, object] = {}
        try:
            update["fairshare"] = self.client.fetch_user_fairshare()
            update["default"] = self.client.fetch_user_default_account()
            update["accounts_error"] = ""
        except Exception as exc:
            update["fairshare"] = {}
            update["default"] = ""
            update["accounts_error"] = str(exc)

        windows: dict[str, object] = {}
        for window, _label in USER_INFO_WINDOWS:
            try:
                windows[window] = self.client.fetch_user_compute_usage(window)
            except Exception:
                windows[window] = "error"
        update["windows"] = windows

        try:
            update["gpfs"] = self.client.fetch_gpfs_quota()
            update["gpfs_error"] = ""
        except Exception as exc:
            update["gpfs"] = None
            update["gpfs_error"] = str(exc)

        try:
            update["gpfs_group_usage"] = self.client.fetch_gpfs_group_usage()
            update["gpfs_group_usage_error"] = ""
        except Exception as exc:
            update["gpfs_group_usage"] = None
            update["gpfs_group_usage_error"] = str(exc)

        with self._info_lock:
            if generation == self._info_generation:
                self._info_data.update(update)
                self._info_data["status"] = "ready"

    def _fetch_info_forecast(self, generation: int) -> None:
        try:
            forecast: object = self.client.fetch_user_fairshare_forecast()
            error = ""
        except Exception as exc:
            forecast = "error"
            error = str(exc)
        with self._info_lock:
            if generation == self._info_generation:
                self._info_data["forecast"] = forecast
                self._info_data["forecast_error"] = error

    def _fetch_info_efficiency(
        self,
        generation: int,
        key: str,
        window: str,
        label: str,
    ) -> None:
        try:
            result: object = self.client.fetch_job_efficiency(window, label)
        except Exception:
            result = "error"
        with self._info_lock:
            if generation == self._info_generation:
                efficiency = self._info_data.get("efficiency")
                if isinstance(efficiency, dict):
                    efficiency[key] = result

    def _info_snapshot(self) -> dict[str, object]:
        with self._info_lock:
            data = dict(self._info_data)
        efficiency = data.get("efficiency")
        if isinstance(efficiency, dict):
            data["efficiency"] = dict(efficiency)
        return data
