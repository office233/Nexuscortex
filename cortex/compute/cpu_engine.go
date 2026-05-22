package compute

// CPUEngine provides a baseline software implementation of the ComputeEngine.
// It is used as a fallback if GPU hardware is unavailable, and as the reference
// truth for unit testing the GPU shaders.
type CPUEngine struct{}

func NewCPUEngine() *CPUEngine {
	return &CPUEngine{}
}

func (e *CPUEngine) Init() error {
	return nil
}

func (e *CPUEngine) Close() {}

func (e *CPUEngine) ForwardSparse(activeIndices []uint32, activeValues []int16, tiles []uint32, bias []int16, tilesPerRow int, outputSize int) ([]int16, error) {
	output := make([]int16, outputSize)
	copy(output, bias)

	for j := 0; j < outputSize; j++ {
		var acc int32
		rowOffset := j * tilesPerRow

		for k, idx := range activeIndices {
			tileIdx := rowOffset + int(idx)/16
			pos := idx % 16

			tile := tiles[tileIdx]
			var sign, mask uint8
			if pos < 8 {
				sign = uint8(tile)
				mask = uint8(tile >> 8)
			} else {
				sign = uint8(tile >> 16)
				mask = uint8(tile >> 24)
				pos -= 8
			}

			bit := uint8(1 << uint(pos))
			if mask&bit != 0 {
				if sign&bit != 0 {
					acc -= int32(activeValues[k]) // weight = -1
				} else {
					acc += int32(activeValues[k]) // weight = +1
				}
			}
			// weight = 0: nothing to do (skip!)
		}

		if acc > 32767 {
			acc = 32767
		} else if acc < -32768 {
			acc = -32768
		}
		output[j] += int16(acc)
	}

	return output, nil
}

func (e *CPUEngine) BatchSDRSimilarity(querySDR []uint32, memorySDRs [][]uint32) ([]uint8, error) {
	results := make([]uint8, len(memorySDRs))

	// Create a fast lookup map for query
	queryMap := make(map[uint32]struct{}, len(querySDR))
	for _, idx := range querySDR {
		queryMap[idx] = struct{}{}
	}

	for i, mem := range memorySDRs {
		var overlap uint8
		for _, idx := range mem {
			if _, ok := queryMap[idx]; ok {
				if overlap < 255 {
					overlap++
				}
			}
		}
		results[i] = overlap
	}

	return results, nil
}
