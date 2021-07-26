import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_pearson = 0
    last_step = 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            model.train()
            feature1, feature2, target = batch.text1, batch.text2, batch.label
            feature1.t_(), feature2.t_() #target.sub_(1)  # batch first, index align
            if args.cuda:
                feature1, feature2, target = feature1.cuda(), feature2.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature1,feature2)
            logit = logit.squeeze(1)
            target = torch.div(target, 5)
            loss = F.mse_loss(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                #corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                #accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f} - best_pearson: {:.4f} - best_step: {}\n'.format(steps, 
                                                        loss.item(),
                                                        best_pearson,
                                                        last_step))
            if steps % args.test_interval == 0:
                pearson = eval(dev_iter, model, args)
                if pearson > best_pearson:
                    best_pearson = pearson
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    avg_loss = 0
    for batch in data_iter:
        feature1, feature2, target = batch.text1, batch.text2, batch.label
        feature1.t_(), feature2.t_() #target.sub_(1)  # batch first, index align
        if args.cuda:
            feature1, feature2, target = feature1.cuda(), feature2.cuda(), target.cuda()

        logit = model(feature1, feature2)
        logit = logit.squeeze(1)
        target = torch.div(target, 5)
        logit_v = logit - torch.mean(logit)
        target_v = target - torch.mean(target)
        pearson = torch.sum(logit_v * target_v) / (torch.sqrt(torch.sum(logit_v ** 2)) * torch.sqrt(torch.sum(target_v ** 2)))
        loss = F.mse_loss(logit, target)

        avg_loss += loss.item()
        #corrects += (torch.max(logit, 1)
                     #[1].view(target.size()).data == target.data).sum()

    
    size = len(data_iter.dataset)
    avg_loss /= size
    print('\nEvaluation - loss: {:.6f} - pearson: {:.4f} \n'.format(avg_loss, pearson))
    return pearson


def predict(text, model, text_field, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    try:
        text1, text2 = text.split('\t')[0], text.split('\t')[1]
    except Exception as e:
        print('error:input is wrong')
    text1 = text_field.preprocess(text1)
    text2 = text_field.preprocess(text2)
    text1 = [[text_field.vocab.stoi[x] for x in text1]]
    text2 = [[text_field.vocab.stoi[x] for x in text2]]
    x = torch.tensor(text1)
    x = autograd.Variable(x)
    y = torch.tensor(text2)
    y = autograd.Variable(y)
    if cuda_flag:
        x = x.cuda()
        y = y.cuda()
    print(x,y)
    output = model(x,y)
    predict = output * 5
    return predict


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
