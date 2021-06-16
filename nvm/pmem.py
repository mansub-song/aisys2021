"""
.. module:: pmem
.. moduleauthor:: Christian S. Perone <christian.perone@gmail.com>

:mod:`pmem` -- low level persistent memory support
==================================================================

.. seealso:: `PMDK libpmem man page
        <http://pmem.io/pmdk/manpages/linux/master/libpmem/libpmem.7.html>`_.
"""
import os
import sys
from _pmem import lib, ffi

#: Create the named file if it does not exist.
FILE_CREATE = 1

#: Ensure that this call creates the file.
FILE_EXCL = 2

#: When creating a file, create a sparse file instead of calling
FILE_SPARSE = 4

#: Create a mapping for an unnamed temporary file.
FILE_TMPFILE = 8


def _coerce_fn(file_name):
    """Return 'char *' compatible file_name on both python2 and python3."""
    if sys.version_info[0] > 2 and hasattr(file_name, 'encode'):
        file_name = file_name.encode(errors='surrogateescape')
    return file_name

class MemoryBuffer(object):
    """A file-like I/O (similar to cStringIO) for persistent mmap'd regions."""

    def __init__(self, buffer_, is_pmem, mapped_len):
        self.buffer = buffer_
        self.is_pmem = is_pmem
        self.mapped_len = mapped_len
        self.size = len(buffer_)
        self.pos = 0

    def __len__(self):
        return self.size

    def _cdata(self):
        return ffi.from_buffer(self.buffer)

    def write(self, data):
        """Write data into the buffer.
        :param data: data to write into the buffer.
        """
        if not data:
            return

        ldata = len(data)
        if (ldata + self.pos) > self.size:
            #print("ldata size: {}".format(ldata))
            #print("self pos: {}".format(self.pos))
            #print("self size: {}".format(self.size))
            raise RuntimeError("Out of range error.")

        new_pos = self.pos + ldata
        self.buffer[self.pos:new_pos] = data
        self.pos = new_pos

    def read(self, size=0):
        """Read data from the buffer.
        :param size: size to read, zero equals to entire buffer size.
        :return: data read.
        """
        if size <= 0:
            if self.pos >= self.size:
                raise EOFError("End of file.")
            data = self.buffer[self.pos:self.size]
            self.pos = self.size
            return data
        else:
            if (self.pos + size) > self.size:
                raise RuntimeError("Out of range error.")
            data = self.buffer[self.pos:self.pos + size]
            self.pos += size
            return data

    def seek(self, pos):
        """Moves the cursor position in the buffer.
        :param pos: the new cursor position
        """
        if pos < 0:
            raise RuntimeError("Negative position.")
        if pos > self.size:
            raise RuntimeError("Out of range error.")
        self.pos = pos

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            if is_pmem(self):
                persist(self)
            else:
                msync(self)
            unmap(self)
        return False


def map_file(file_name, file_size, flags, mode):
    """mmap it using pmem.
        Given a path, this function creates a new read/write mapping for the named file
    
    :param file_name: The file name to use.
    :param file_size: the size to allocate

    :return: The mapping, an exception will rise in case
             of error.
    """
    ret_mappend_len = ffi.new("size_t *")
    ret_is_pmem = ffi.new("int *")
    ret = lib.pmem_map_file(_coerce_fn(file_name), file_size, flags, mode,
                            ret_mappend_len, ret_is_pmem)
    
    if ret == ffi.NULL:
        raise RuntimeError(os.strerror(ffi.errno))

    ret_mapped_len = ret_mappend_len[0]
    ret_is_pmem = bool(ret_is_pmem[0])

    cast = ffi.buffer(ret, file_size)
    return MemoryBuffer(cast, ret_is_pmem, ret_mapped_len)

def is_pmem(memory_buffer):
    """Return true if entire range is persistent memory.

    :return: True if the entire range is persistent memory, False otherwise.
    """
    cdata = memory_buffer._cdata()
    ret = lib.pmem_is_pmem(cdata, len(memory_buffer))
    return bool(ret)

def persist(memory_buffer):
    """Make any cached changes to a range of pmem persistent.

    :param memory_buffer: the MemoryBuffer object.
    """
    cdata = memory_buffer._cdata()
    lib.pmem_persist(cdata, len(memory_buffer))
