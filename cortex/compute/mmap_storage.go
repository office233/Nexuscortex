package compute

import (
	"fmt"
	"os"
	"reflect"
	"unsafe"

	"github.com/edsrzf/mmap-go"
)

type MmapStorage struct {
	file *os.File
	mmap mmap.MMap
	path string
}

// NewMmapStorage maps an existing file or creates a new one of the given size (in bytes).
// It returns a slice of uint32 that points directly to the file data.
func NewMmapStorage(path string, sizeBytes int) (*MmapStorage, []uint32, error) {
	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open file %s: %w", path, err)
	}

	info, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, nil, fmt.Errorf("failed to stat file %s: %w", path, err)
	}

	if info.Size() < int64(sizeBytes) {
		if err := file.Truncate(int64(sizeBytes)); err != nil {
			file.Close()
			return nil, nil, fmt.Errorf("failed to truncate file %s: %w", path, err)
		}
	}

	mapped, err := mmap.Map(file, mmap.RDWR, 0)
	if err != nil {
		file.Close()
		return nil, nil, fmt.Errorf("failed to mmap file %s: %w", path, err)
	}

	// Cast the mapped bytes to []uint32
	var uint32Slice []uint32
	header := (*reflect.SliceHeader)(unsafe.Pointer(&uint32Slice))
	header.Data = uintptr(unsafe.Pointer(&mapped[0]))
	header.Len = len(mapped) / 4
	header.Cap = len(mapped) / 4

	return &MmapStorage{
		file: file,
		mmap: mapped,
		path: path,
	}, uint32Slice, nil
}

func (s *MmapStorage) Sync() error {
	if s.mmap != nil {
		return s.mmap.Flush()
	}
	return nil
}

func (s *MmapStorage) Close() error {
	var err1, err2 error
	if s.mmap != nil {
		err1 = s.mmap.Unmap()
		s.mmap = nil
	}
	if s.file != nil {
		err2 = s.file.Close()
		s.file = nil
	}
	if err1 != nil {
		return err1
	}
	return err2
}
