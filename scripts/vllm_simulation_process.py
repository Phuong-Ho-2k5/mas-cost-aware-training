"""Own the whole Windows simulation tree, including workers orphaned by its CLI."""
import ctypes
from ctypes import wintypes
import os
import subprocess
import time


class BasicLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                ("flags", wintypes.DWORD), ("minimum_working_set", ctypes.c_size_t),
                ("maximum_working_set", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]


class IOCounts(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("reads", "writes", "other", "read_bytes", "write_bytes", "other_bytes")]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", BasicLimits), ("io", IOCounts), ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t),
                ("peak_job_memory", ctypes.c_size_t)]


class Accounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_longlong) for name in ("user", "kernel", "period_user", "period_kernel")] + [
        (name, wintypes.DWORD) for name in ("page_faults", "total_processes", "active_processes", "terminated")]


class OwnedSimulation:
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("This owned simulation runner requires Windows Job Objects")
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.api.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.api.CloseHandle(self.handle)
            self.handle = None
            raise ctypes.WinError(ctypes.get_last_error())

    def start(self, command, **kwargs):
        # Assign before executing any user code, so no worker can escape the job.
        child = subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW | 0x4, **kwargs)
        try:
            if not self.api.AssignProcessToJobObject(self.handle, int(child._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            nt = ctypes.WinDLL("ntdll")
            nt.NtResumeProcess.argtypes = [wintypes.HANDLE]
            if nt.NtResumeProcess(int(child._handle)) != 0:
                raise RuntimeError("Could not resume the owned simulation process")
            return child
        except BaseException:
            child.kill()
            child.wait(5)
            self.stop()
            raise

    def active_processes(self):
        counts = Accounting()
        if not self.api.QueryInformationJobObject(self.handle, 1, ctypes.byref(counts), ctypes.sizeof(counts), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return counts.active_processes

    def stop(self, timeout=5):
        if self.handle is None:
            return {"active_processes": 0, "container": "Windows Job Object"}
        try:
            if not self.api.TerminateJobObject(self.handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            deadline = time.monotonic() + timeout
            while self.active_processes() and time.monotonic() < deadline:
                time.sleep(0.02)
            remaining = self.active_processes()
            if remaining:
                raise RuntimeError(f"Simulation job still has {remaining} active survivors")
            return {"active_processes": 0, "container": "Windows Job Object"}
        finally:
            self.api.CloseHandle(self.handle)
            self.handle = None
