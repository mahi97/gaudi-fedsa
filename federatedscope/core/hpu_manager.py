import os


def check_hpus():
    if not 'HABANALABS AIPs' in os.popen('hl-smi -h').read():
        print("'hl-smi' tool not found.")
        return False
    return True


class HPUManager():
    """
    To automatic allocate the hpu, which returns the hpu with the largest
    free memory rate, unless the specified_device has been set up
    When hpus is unavailable, return 'cpu';
    The implementation of HPUManager is referred to
    https://github.com/QuantumLiu/tf_gpu_manager
    """
    def __init__(self, hpu_available=False, specified_device=-1):
        self.hpu_avaiable = hpu_available and check_hpus()
        self.specified_device = specified_device
        if self.hpu_avaiable:
            self.hpus = self._query_hpus()
            for hpu in self.hpus:
                hpu['allocated'] = False
        else:
            self.hpus = None

    def _sort_by_memory(self, hpus, by_size=False):
        if by_size:
            return sorted(hpus, key=lambda d: d['memory.free'], reverse=True)
        else:
            print('Sorted by free memory rate')
            return sorted(
                hpus,
                key=lambda d: float(d['memory.free']) / d['memory.total'],
                reverse=True)

    def _query_hpus(self):
        args = ['index', 'name', 'memory.free', 'memory.total']
        cmd = 'hl-smi --query-aip={} --format=csv,noheader'.format(
            ','.join(args))
        results = os.popen(cmd).readlines()
        return [self._parse(line, args) for line in results]

    def _parse(self, line, args):
        numberic_args = ['memory.free', 'memory.total']
        to_numberic = lambda v: float(v.upper().strip().replace('MIB', '').
                                      replace('W', ''))
        process = lambda k, v: (int(to_numberic(v))
                                if k in numberic_args else v.strip())
        return {
            k: process(k, v)
            for k, v in zip(args,
                            line.strip().split(','))
        }

    def auto_choice(self):
        """
        To allocate a device
        """
        if self.hpus is None:
            return 'cpu'
        elif self.specified_device >= 0:
            # allow users to specify the device
            return 'hpu:{}'.format(self.specified_device)
        else:
            for old_infos, new_infos in zip(self.hpus, self._query_hpus()):
                old_infos.update(new_infos)
            unallocated_hpus = [
                hpu for hpu in self.hpus if not hpu['allocated']
            ]
            if len(unallocated_hpus) == 0:
                # reset when all hpus have been allocated
                unallocated_hpus = self.hpus
                for hpu in self.hpus:
                    hpu['allocated'] = False

            chosen_hpu = self._sort_by_memory(unallocated_hpus, True)[0]
            chosen_hpu['allocated'] = True
            index = chosen_hpu['index']
            return 'hpu:{:s}'.format(index)


# for testing
if __name__ == '__main__':

    hpu_manager = HPUManager(hpu_available=True, specified_device=0)
    for i in range(20):
        print(hpu_manager.auto_choice())
