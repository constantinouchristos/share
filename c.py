import torch
import torch.nn as nn
import torch.distributed as dist

# taken from https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
      
      
      def create_indexes_contrastive(labels_b,
                               labels_back
                              ):
    """
    labels_b   : (np.array)
    labels_back: (np.array)
    
    """
    
    indexes_dict = {i:j for i,j in enumerate(labels_b)}

    already_used = set()

    set_conistis = [0,1,2,3]

    # group indexes label
    indexes_slected = {}

    cc = 0
    while len(already_used) < len(indexes_dict):

        indexes_slected[cc] = []

        for c in set_conistis:
            break_c_loop = False
            for j,k in indexes_dict.items():

                if k == c and j not in already_used and not break_c_loop:
                    indexes_slected[cc].append(j)
                    already_used.add(j)
                    break_c_loop = True
        cc += 1
        
    indexes_slected = {i:j for i,j in indexes_slected.items() if len(j)>=2}
    
    # indexes from backgrooung labels
    matching_indx_bc = {}

    for j in indexes_slected:
        
        matching_indx_bc[j] = []
        temp_l_labels = labels_b[indexes_slected[j]]
        
#         print(temp_l_labels)

#         print(f'indexes original for {j}, {indexes_slected[j]}, labels: {temp_l_labels} ')

        for c in temp_l_labels:

            index_where = np.where(labels_back == c)[0]
#             print(f'index_where: {index_where}')
            random_choice = np.random.choice(index_where)
            matching_indx_bc[j].append(random_choice)
            
#         print('chosen_indexes :',matching_indx_bc[j])
#         print('labels back:', labels_back.cpu().numpy()[matching_indx_bc[j]])
        
    return indexes_slected,matching_indx_bc
    
    
loss_to_average = 0
total_contrastive = 0

for ii in indexes_slected:

    indexes_in = indexes_slected[ii]
    indexes_back = matching_indx_bc[ii]

    temp_projections_in = projections_in[indexes_in,:]
    temp_projections_back = projections_back[indexes_back,:]

    temp_b_size = temp_projections_in.shape[0]

    contrastive_critirion = NT_Xent(batch_size=temp_b_size,temperature=0.7,world_size=1)
    contrastive_loss = contrastive_critirion(temp_projections_in,temp_projections_back)

    total_contrastive += contrastive_loss

    loss_to_average += 1

total_contrastive = total_contrastive / loss_to_average
    
    
