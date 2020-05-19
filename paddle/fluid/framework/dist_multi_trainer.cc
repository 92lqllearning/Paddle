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
#include "io/fs.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/trainer.h"

namespace paddle {
namespace framework {

void DistMultiTrainer::Initialize(const TrainerDesc &trainer_desc,
                                  Dataset *dataset) {
  thread_num_ = trainer_desc.thread_num();
  SetDataset(dataset);

  ParseDumpConfig(trainer_desc);
  mpi_rank_ = trainer_desc.mpi_rank() / 2;
  mpi_size_ = trainer_desc.mpi_size() / 2;
  dump_file_num_ = trainer_desc.dump_file_num();
  const std::vector<paddle::framework::DataFeed *> readers =
      dataset->GetReaders();

  thread_num_ = readers.size();
  workers_.resize(thread_num_);
  for (int i = 0; i < trainer_desc.downpour_param().stat_var_names_size();
       i++) {
    need_merge_var_names_.push_back(
        trainer_desc.downpour_param().stat_var_names(i));
  }

  for (int i = 0; i < thread_num_; ++i) {
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    workers_[i]->SetDeviceIndex(i);
    workers_[i]->SetDataFeed(readers[i]);
    workers_[i]->SetNeedDumpField(need_dump_field_);
    workers_[i]->SetNeedDumpParam(need_dump_param_);
    workers_[i]->SetDumpFieldVector(dump_fields_);
    workers_[i]->SetDumpParamVector(dump_param_);
    workers_[i]->InitRandomDumpConfig(trainer_desc);
    workers_[i]->Initialize(trainer_desc);
  }

  VLOG(3) << "going to initialize pull dense worker";
  pull_dense_worker_ = PullDenseWorker::GetInstance();
  pull_dense_worker_->Initialize(trainer_desc);
  VLOG(3) << "initialize pull dense worker";
  SetDebug(trainer_desc.debug());
}

void DistMultiTrainer::InitDumpEnv() {
  queue_ = paddle::framework::MakeChannel<std::string>();
  for (int i = 0; i < thread_num_; ++i) {
    workers_[i]->SetChannelWriter(queue_.get());
  }
  dump_thread_num_ = 1;
  if (dump_file_num_ > mpi_size_) {
    dump_thread_num_ = dump_file_num_ / mpi_size_;
    if (dump_file_num_ % mpi_size_ > mpi_rank_) {
      dump_thread_num_ += 1;
    }
  }
  for (int i = 0; i < dump_thread_num_; i++) {
    dump_thread_.push_back(
        std::thread(std::bind(&TrainerBase::DumpWork, this, i)));
  }
}

void DistMultiTrainer::InitOtherEnv(const ProgramDesc &main_program) {
  if (need_dump_field_) {
    InitDumpEnv();
  }
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
  for (auto &th : threads_) {
    th.join();
  }
  for (int i = 0; i < need_merge_var_names_.size(); i++) {
    Variable *root_var = root_scope_->FindVar(need_merge_var_names_[i]);
    if (root_var == nullptr) {
      continue;
    }
    LoDTensor *root_tensor = root_var->GetMutable<LoDTensor>();
    for (int j = 1; j < thread_num_; j++) {
      Scope *cur_thread_scope = workers_[j]->GetThreadScope();
      Variable *thread_var =
          cur_thread_scope->FindVar(need_merge_var_names_[i]);
      LoDTensor *thread_tensor = thread_var->GetMutable<LoDTensor>();
      if (root_tensor->numel() != thread_tensor->numel()) {
        continue;
      }
#define MergeCallback(cpp_type, proto_type)                                    \
  do {                                                                         \
    if (root_tensor->type() == proto_type) {                                   \
      if (thread_tensor->type() != proto_type) {                               \
        VLOG(0) << "Error: thread id=" << j << ", need_merge_var_names_[" << i \
                << "] " << need_merge_var_names_[i]                            \
                << ", root tensor type=" << root_tensor->type()                \
                << ", thread tensor type=" << thread_tensor->type();           \
        exit(-1);                                                              \
      }                                                                        \
      MergeToRootScope<cpp_type>(root_tensor, thread_tensor);                  \
    }                                                                          \
  } while (0)
      _ForEachDataType_(MergeCallback);
    }
  }

  if (need_dump_field_) {
    FinalizeDumpEnv();
  }
  pull_dense_worker_->Stop();
  root_scope_->DropKids();

  // flush local client push queue
  auto fleet_ptr_ = FleetWrapper::GetInstance();
  fleet_ptr_->ClientFlush();
}

template <typename T>
void DistMultiTrainer::MergeToRootScope(LoDTensor *root_tensor,
                                        LoDTensor *tensor) {
  T *root_data = root_tensor->data<T>();
  T *data = tensor->data<T>();
  for (int i = 0; i < tensor->numel(); i++) {
    root_data[i] += data[i];
  }
}
}  // end namespace framework
}  // end namespace paddle
