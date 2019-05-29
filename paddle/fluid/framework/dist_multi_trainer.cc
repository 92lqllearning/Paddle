/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <vector>
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/trainer.h"

namespace paddle {
namespace framework {

void DistMultiTrainer::Initialize(const TrainerDesc& trainer_desc,
                                  Dataset* dataset_ptr) {
  thread_num_ = trainer_desc.thread_num();
  SetDataset(dataset);

//  dataset->CreateReaders();
//  const std::vector<std::shared_ptr<paddle::framework::DataFeed>> readers =
  
/*  dataset_ = std::make_shared<paddle::framework::MultiSlotDataset>();

  dataset_->SetThreadNum(12);
  dataset_->SetHdfsConfig("afs://xingtian.afs.baidu.com:9902","mlarch,Fv1M87");
  dataset_->SetFileList(std::vector<std::string>{
          "afs:/user/feed/mlarch/sequence_generator/heqiaozhi/feasign/20180920/00/part-00000.gz",
          "afs:/user/feed/mlarch/sequence_generator/heqiaozhi/feasign/20180920/00/part-00001.gz",
          "afs:/user/feed/mlarch/sequence_generator/heqiaozhi/feasign/20180920/00/part-00002.gz",
          "afs:/user/feed/mlarch/sequence_generator/heqiaozhi/feasign/20180920/00/part-00003.gz",
          "afs:/user/feed/mlarch/sequence_generator/heqiaozhi/feasign/20180920/00/part-00004.gz",
          "afs:/user/feed/mlarch/sequence_generator/heqiaozhi/feasign/20180920/00/part-00005.gz",
          "afs:/user/feed/mlarch/sequence_generator/heqiaozhi/feasign/20180920/00/part-00006.gz",
          "afs:/user/feed/mlarch/sequence_generator/heqiaozhi/feasign/20180920/00/part-00007.gz",
          "afs:/user/feed/mlarch/sequence_generator/heqiaozhi/feasign/20180920/00/part-00008.gz",
          "afs:/user/feed/mlarch/sequence_generator/heqiaozhi/feasign/20180920/00/part-00009.gz",
          "afs:/user/feed/mlarch/sequence_generator/heqiaozhi/feasign/20180920/00/part-00010.gz",
          "afs:/user/feed/mlarch/sequence_generator/heqiaozhi/feasign/20180920/00/part-00011.gz"
          });

  std::string f = "/home/disk6/xujiaqi/mycode/paddle_fkfkfk/Paddle/build_withoutpslib/paddle/fluid/framework/data_feed.prottxt";
  std::ifstream t(f);
  std::stringstream buffer;
  buffer << t.rdbuf();
  std::string contents(buffer.str());
  dataset_->SetDataFeedDesc(contents);
  
  dataset_->SetFleetSendBatchSize(800);

  dataset_->LoadIntoMemory();
*/ 
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset_->GetReaders();

  thread_num_ = readers.size();
  workers_.resize(thread_num_);

  for (int i = 0; i < thread_num_; ++i) {
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    workers_[i]->SetDeviceIndex(i);
    workers_[i]->SetDataFeed(readers[i]);
    workers_[i]->Initialize(trainer_desc);
  }

  VLOG(3) << "going to initialize pull dense worker";
  pull_dense_worker_ = PullDenseWorker::GetInstance();
  pull_dense_worker_->Initialize(trainer_desc);
  VLOG(3) << "initialize pull dense worker";
  SetDebug(trainer_desc.debug());
}

void DistMultiTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  pull_dense_worker_->SetRootScope(root_scope_);
  pull_dense_worker_->Start();
  VLOG(3) << "init other env done.";
}

void DistMultiTrainer::Run() {
  for (int thidx = 0; thidx < thread_num_; ++thidx) {
    if (!debug_) {
      threads_.push_back(
          std::thread(&DeviceWorker::TrainFiles, workers_[thidx].get()));
    } else {
      threads_.push_back(std::thread(&DeviceWorker::TrainFilesWithProfiler,
                                     workers_[thidx].get()));
    }
  }
}

void DistMultiTrainer::Finalize() {
  for (auto& th : threads_) {
    th.join();
  }
  pull_dense_worker_->Stop();
//  dataset_ptr_->DestroyReaders();
  root_scope_->DropKids();
}

}  // end namespace framework
}  // end namespace paddle
