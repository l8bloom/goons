package services

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
	"golang.org/x/sys/unix"
)

var (
	// SD_API const char* sd_get_system_info();
	getSystemInfo ffi.Fun
)

func loadSystemRoutines(lib ffi.Lib) error {
	var err error
	if getSystemInfo, err = lib.Prep("sd_get_system_info", &ffi.TypePointer); err != nil {
		return loadError("sd_get_system_info", err)
	}

	return nil
}

func GetSystemInfo() string {
	var systemInfo *byte

	getSystemInfo.Call(unsafe.Pointer(&systemInfo))
	if systemInfo == nil {
		return ""
	}

	return unix.BytePtrToString(systemInfo)
}
