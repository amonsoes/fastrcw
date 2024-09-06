import torch
import csv
import torch.optim as optim
import matplotlib.pyplot as plt

from random import randint
from diff_jpeg import diff_jpeg_coding
from src.adversarial.custom_cw import CW
from torchvision.io import encode_jpeg, decode_image

#############################
#
# FastRCW code written by A. S.
# Base Provided by Carlini et al. (2017)
#
#############################



class RCW(CW):
    
    def __init__(self, 
                model, 
                model_trms, 
                rcw_comp_lower_bound, 
                rcw_beta, 
                *args, 
                **kwargs):
        """Robust CW employs a differentiable approximation
        in the constrained optimization in the form of T
        to improve the robustness in compression scenarios.
        
        ||x - x_hat||_2 + b * f(JPEG(x_hat, q), theta)
        
        jpeg_quality argument should be in range [0,1]
        they will be internally projected to 255
        and after compression back to 1.0
        
        self.compress(x) should be applied directly on image 
        CQE(Compression Quality Estimation) is a binary search algorithm
        to find the quality setting of a JPEG algorithm
        
        """
        super().__init__(model, model_trms, *args, **kwargs)
        self.compression_lower_bound = rcw_comp_lower_bound
        self.original_c = self.c
        self.cqe_log = {}
        
        
    def compress(self, img, jpeg_quality):
        img = img * 255
        compressed =  diff_jpeg_coding(image_rgb=img, jpeg_quality=jpeg_quality)
        return (compressed / 255).clip(min=0., max=1.)

    def compress_for_q_search(self, img, jpeg_quality):
        img = (img * 255).to(torch.uint8)
        compressed = encode_jpeg(img.squeeze(0), quality=jpeg_quality)
        compressed_img = decode_image(compressed)
        compressed_img = compressed_img / 255
        return compressed_img.clip(min=0., max=1.).unsqueeze(0)
    
    def compression_quality_estimation(self, cqe_image, ref_image, acceptance_threshold=1.0,step_size=1):
        """
        determines JPEG compression quality based on binary search.
        
        cqe_image: output of JPEG algo we want to compare against
        ref_image: uncompressed image used to find the right compression val
        schedule: constant thta scales down the step size. Gets smaller as n_steps increases
        q: JPEG quality setting we want to change to find the true compression setting
        exploration_step: keep exploring direction even if previous step resulted in worse l2
        
        """
        print('######## INIT JPEG SETTING SEARCH ####### \n')
        ref_image = ref_image.to('cpu')
        cqe_image = cqe_image.to('cpu')
        direction = -1
        schedule = 1.0
        q = randint(1,99)
        #q = 98
        current_l2 = 1e07
        best_l2 = current_l2
        best_q = q
        break_criterion = 0
        n_steps = 0
        # keep exploring direction even if previous step resulted in worse l2
        exploration_steps = 0

        compressed_img = self.compress_for_q_search(ref_image, jpeg_quality=torch.tensor([q]).to(self.device))
        last_l2 = (compressed_img - cqe_image).pow(2).sum().sqrt()
        
        while current_l2 > acceptance_threshold:
            q += direction * max(int(schedule * step_size),1)
            q = min(max(q, 1), 99)
            compressed_img = self.compress_for_q_search(ref_image, jpeg_quality=torch.tensor([q]).to(self.device))
            current_l2 = (compressed_img - cqe_image).pow(2).sum().sqrt()
            if current_l2 >= last_l2:
                exploration_steps += 1
                if exploration_steps > 3:
                    direction *= -1
                    exploration_steps = 0
            elif current_l2 < best_l2:
                best_q = q
                break_criterion = 0
                best_l2 = current_l2
                exploration_steps = 0
            step_size = int(current_l2)
            last_l2 = current_l2
            schedule *= 0.95
            n_steps += 1
            if break_criterion >= 10 or n_steps >=100:
                # best q has not changed for 4 iterations
                break
            break_criterion += 1
            print(f'step : {n_steps}, q : {q}, current_l2 : {current_l2}, best q : {best_q}\n')
        print(f'BEST q FOUND: {best_q}')
        return best_q
        
    def get_robustness_loss(self, adv_images, labels, target_labels):
    
        #random_quality = random.randint(self.compression_lower_bound, 99)
        compressed_img = self.compress(adv_images, jpeg_quality=torch.tensor([self.determined_quality]).to(self.device))
        
        #outputs = self.get_outputs(adv_images)
        compressed_outputs = self.get_outputs(compressed_img) 
        
        if self.targeted:
            f_compressed = self.f(compressed_outputs, target_labels).sum()
        else:
            f_compressed = self.f(compressed_outputs, labels).sum()
        return f_compressed, compressed_outputs
    
    def forward(self, images, labels, cqe_image):
        r"""
        Overridden.
        """
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255
        
        self.c = self.original_c
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        self.determined_quality = self.compression_quality_estimation(cqe_image=cqe_image, ref_image=images)
        if str(self.determined_quality) in self.cqe_log.keys():
            self.cqe_log[str(self.determined_quality)] += 1
        else:
            self.cqe_log[str(self.determined_quality)] = 1

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        
        if self.use_attack_mask:
            attack_mask = self.hpf_masker(images, 
                                        labels, 
                                        model=self.model, 
                                        model_trms=self.model_trms, 
                                        loss=self.surr_model_loss,
                                        invert_mask=True)
        else:
            attack_mask = 1 # multiplication results in identity

        #implement outer step like in perccw

        # set the lower and upper bounds accordingly
        lower_bound = torch.zeros(1, device=self.device)
        CONST = torch.full((1,), self.c, device=self.device)
        upper_bound = torch.full((1,), 1e10, device=self.device)
        
        # set markers for inter-c-run comparison
        best_adv_images_from_starts = images.clone().detach()
        best_cost_from_starts = 1e10
        adv_found = torch.tensor([0])
        
        for outer_step in range(self.n_samples):    
        # search to get optimal c
            print(f'step in binary search for constant c NR:{outer_step}, c:{self.c}\n')
        
            # w = torch.zeros_like(images).detach() # Requires 2x times    
            w = self.inverse_tanh_space(images).detach()
            w.requires_grad = True

            best_adv_images = images.clone().detach()
            #best_iq = 1e10*torch.ones((len(images))).to(self.device)
            best_cost = 1e10*torch.ones((len(images))).to(self.device)
            #dim = len(images.shape)

            self.optimizer = optim.Adam([w], lr=self.lr)
            self.optimizer.zero_grad()

            for step in range(self.steps):
                    
                # Get adversarial images
                adv_images = self.tanh_space(w)

                # Calculate image quality loss
                iq_loss, current_iq_loss = self.get_iq_loss(adv_images, images, attack_mask)

                # Calculate adversarial loss including robustness loss
                ro_loss, comp_outputs = self.get_robustness_loss(adv_images, labels, target_labels)

                cost = iq_loss + self.c*ro_loss
                
                # Update adversarial images
                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                # filter out images that get either correct predictions or non-decreasing loss, 
                # i.e., only images that are not target and loss-decreasing are left 
                # check if adv_images were classified incorrectly

                _, pre = torch.max(comp_outputs.detach(), 1)
                is_adversarial = (pre == target_labels).float()            
                is_lower_in_cost = (best_cost > cost.detach())
                
                mask =  is_adversarial * is_lower_in_cost
                #mask = mask.view([-1]+[1]*(dim-1))

                best_cost = mask*cost.detach() + (1-mask)*best_cost
                best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images
                # either the current adv_images is the new best_adv_images or the old one depending on mask
                
                print(f'\n{step} - iq_loss: {current_iq_loss.item()}, r_loss: {ro_loss.item()} cost: {cost.item()}')     

            # set the best output of run as best adv if (1) an adv was found (2) the cost is lower than the one from the last starts
            adv_found_in_run = torch.any(best_adv_images != images) # could only be different if one output was adv during optim
            is_lower_than_all_starts = (best_cost_from_starts > best_cost).float()
            mask = adv_found_in_run * is_lower_than_all_starts
            best_adv_images_from_starts = mask*best_adv_images + (1-mask)*best_adv_images_from_starts
            best_cost_from_starts = mask*best_cost + (1-mask)*best_cost_from_starts

            # adjust the constant as needed
            adv_found = adv_found_in_run
            upper_bound[adv_found] = torch.min(upper_bound[adv_found], CONST[adv_found])
            adv_not_found = ~adv_found
            lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], CONST[adv_not_found])
            is_smaller = upper_bound < 1e9
            CONST[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
            CONST[(~is_smaller) * adv_not_found] *= 10
            self.c = CONST.item()
        
            # from perccw.py
            ######################   

        if self.protocol_file:
            self.write_to_protocol_dir(iq_loss, ro_loss, cost)
            self.write_runtime_to_protocol_dir()
        return (best_adv_images, target_labels)

    def log_cqe_results(self, path):
        with open(path, 'w') as log_file:
            csv_obj = csv.writer(log_file)
            csv_obj.writerow(['QUALITY', 'OCCURENCES'])
            for k,v in self.cqe_log.items():
                csv_obj.writerow([k,v])
        

        q_setting, _ = max(self.cqe_log.items(), key= lambda x: x[1])
        q_setting = int(q_setting)
        # we are assuming that the desired setting is the maximum chosen value
        #sorted_items = sorted(items, key=lambda x: x[0])
        x_list = list(range(q_setting - 10, q_setting + 10 + 1))
        y_list = [self.cqe_log.get(str(x), 0) for x in x_list]
        
        y_sum = sum(y_list)
        y_list = [y / y_sum for y in y_list]
        
        plt.figure(figsize=(8, 6))
        plt.bar(x_list, y_list, color='skyblue')
        plt.xticks(x_list)
        plt.xlabel('Compression Values')
        plt.ylabel('y')
        plt.title('Compression Adaptation Results')
        
        path_dir = '/'.join(path.split('/')[:-1]) + '/' + 'cqe_fig.png'
        plt.savefig(path_dir)
        plt.show()
        