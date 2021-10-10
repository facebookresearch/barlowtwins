import torch

'''
Manages launching the job eithere as multi- process or single
'''

def launch_job(args, func, init_method=None, ):
    """
    Run 'func' on one or more GPUs, specified in cfg
    """
    args.ngpus_per_node = torch.cuda.device_count()
    args.rank = 0
    args.world_size = max( args.ngpus_per_node, 1)
    if args.ngpus_per_node > 0:
        print("Launching job on {} gpus".format( args.ngpus_per_node))
        torch.multiprocessing.spawn(run_multi_process, args=(func, args,), nprocs=args.ngpus_per_node)
    else:
        func(0)

def run_multi_process(gpu, func, args,):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend=args.backend, init_method=None,
        world_size=args.ngpus_per_node, rank=args.rank)

    torch.cuda.set_device(gpu)

    func(gpu)

