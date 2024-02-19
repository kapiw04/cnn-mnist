import time
import sys

time_start = time.time()

def eta(epoch, i, epochs, dataloader, time_start=time_start, include_hours=False):
    time_end = time.time()
    avg_time_per_batch = (time_end - time_start) / (i + 1)
    time_to_finish_current_epoch = avg_time_per_batch * (len(dataloader) - (i + 1))
    time_for_remaining_epochs = avg_time_per_batch * len(dataloader) * (epochs - epoch - 1)
    eta = time_to_finish_current_epoch + time_for_remaining_epochs
    time_formatted = time.strftime('%Hh %Mm %Ss', time.gmtime(eta)) if include_hours else time.strftime('%Mm %Ss', time.gmtime(eta))
    sys.stdout.write(f'\rEpoch: {epoch + 1}/{epochs}, {i + 1}/{len(dataloader)}, ETA: {time_formatted}')
    sys.stdout.flush()