# Compile storage and its accessors directly; exact retired charges remain Python objects.
cdef class _Labels:
    cdef public object ints, floats
    cdef public long long[:, ::1] int_view
    cdef public double[:, ::1] float_view
    cdef public list retired, floors
    cdef public Py_ssize_t size
    cdef public bint compiled, native
    cpdef Py_ssize_t add(
        self, Py_ssize_t node, Py_ssize_t parent, Py_ssize_t context,
        Py_ssize_t paid, Py_ssize_t sequence, bint terminal, bint first,
        double stamp, double length, double positive, double canonical_length,
        object retired, double floor=*,
    )
    cpdef tuple path(self, Py_ssize_t label, tuple cells)


cdef class _RowTable:
    cdef public tuple rows
    cdef public dict ids, spaces
    cdef public object blocked, positive, offsets, steps, row_ids, marks, generation


cdef extern from *:
    """
    #include <stdint.h>
    #include <string.h>
    static uint64_t _zeta_state_bits(double value) {
        uint64_t bits;
        if (value == 0.0) return 0;
        memcpy(&bits, &value, sizeof(bits));
        return bits;
    }
    static uint64_t _zeta_state_mix(uint64_t first, uint64_t second) {
        second += UINT64_C(0x9e3779b97f4a7c15);
        second = (second ^ (second >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        second = (second ^ (second >> 27)) * UINT64_C(0x94d049bb133111eb);
        return (first ^ second ^ (second >> 31)) * UINT64_C(0x9e3779b97f4a7c15);
    }
    """
    unsigned long long _zeta_state_bits(double value) noexcept
    unsigned long long _zeta_state_mix(unsigned long long first, unsigned long long second) noexcept


cdef class _StateIndex:
    cdef public Py_ssize_t size
    cdef long long[::1] slots
    cdef long long[:, ::1] integers
    cdef double[:, ::1] clocks
    cdef unsigned long long[::1] hashes
    cdef void _grow(self) except *
    cdef Py_ssize_t _intern(
        self, long long kind, long long node, long long context, double stamp,
        double length, double canonical, long long roles, long long sequence,
        unsigned long long hashed,
    ) except -1
    cpdef tuple keys(
        self, long long node, long long context, double stamp, double length,
        double canonical, bint terminal, bint first, long long sequence,
        bint broad, bint shorter,
    )
