# -*- coding: utf-8 -*-

import logging
import multiprocessing
import codecs
import itertools

from .common import get_input_files

logger = logging.getLogger(__name__)


class Dataset(object):
    def __iter__(self):
        """Iterate over Dataset, **should be overridden in inheritor class**.
        """
        raise NotImplementedError()

    def __len__(self):
        """Get size of dataset, **should be overridden in inheritor class**.
        """
        raise NotImplementedError()


class TextDataset(Dataset):
    def __init__(self, input_files, encoding='utf-8', pipeline=None,
                 interleave=False, n_workers=1):
        self.length = None
        self.input_files = get_input_files(input_files)
        logger.debug('self.input_files: %s' % self.input_files)
        self.encoding = encoding
        if pipeline is not None:
            self.pipeline = pipeline if isinstance(pipeline, (list, tuple)) \
                             else [pipeline]
        else:
            self.pipeline = None
        if n_workers <= 0:
            self.n_workers = multiprocessing.cpu_count() + n_workers
        else:
            self.n_workers = n_workers
        self.interleave = interleave

    def run_pipeline(self, data):
        if self.pipeline is None:
            return data
        for proc in self.pipeline:
            try:
                data = proc(data)
                if data is None:
                    break
            except Exception as ex:
                logger.warn('run_pipeline error %s' % ex)
                return None
        return data

    def get_stream(self):
        if not self.interleave:
            for fn in self.input_files:
                with codecs.open(fn, 'rb', self.encoding) as fin:
                    for line in fin:
                        yield line.strip('\r\n')
        else:
            input_fins = [codecs.open(fn, 'rb', self.encoding)
                          for fn in self.input_files]
            for lines in itertools.izip_longest(input_fins):
                for line in lines:
                    if line is not None:
                        yield line.strip('\r\n')
            for fin in input_fins:
                fin.close()

    def reader_proc(self, input_queue, n_workers):
        for line in self.get_stream():
            try:
                input_queue.put(line)
            except Exception as ex:
                logger.warn('reader_proc error %s, line: %s' % (ex, line))
                continue
        for i in xrange(n_workers):
            input_queue.put(None)

    def worker_proc(self, input_queue, output_queue):
        while True:
            try:
                line = input_queue.get()
                if line is None:
                    break
                data = self.run_pipeline(line)
                if data is None:
                    continue
                output_queue.put(data)
            except Exception as ex:
                logger.warn('worker_proc error: %s' % ex)
                continue
        output_queue.put(None)

    def __len__(self):
        if self.length is None:
            self.length = sum(1 for _ in self.__iter__())
        return self.length

    def __iter__(self):
        if self.n_workers == 1:
            for line in self.get_stream():
                data = self.run_pipeline(line)
                if data is not None:
                    yield data
        else:
            input_queue = multiprocessing.Queue(1000 * self.n_workers)
            output_queue = multiprocessing.Queue(1000 * self.n_workers)
            workers = []
            for i in xrange(self.n_workers):
                worker = multiprocessing.Process(
                    target=self.worker_proc, args=(input_queue, output_queue))
                worker.start()
                workers.append(worker)
                logger.debug('Dataset worker %s started.' % worker.name)
            reader = multiprocessing.Process(
                target=self.reader_proc, args=(input_queue, self.n_workers))
            reader.start()
            n_done = 0
            while True:
                try:
                    data = output_queue.get()
                    if data is None:
                        n_done += 1
                        if n_done == self.n_workers:
                            break
                    else:
                        yield data
                except Exception as ex:
                    logger.warn('__iter__ error: %s' % ex)
                continue
            reader.join()
            for worker in workers:
                worker.join()
